"""
vcond.py

PyTorch Module defining the Voltron `V-Cond` variant (single-frame with language-conditioning). In general, follows the
MAE recipe, with the architectural modifications described in the paper:
    - RMSNorm, for stability/performance ("Do Transformer Modifications Transfer...")
    - SwishGLU activations in the Attention Block Feed-Forward MLP (gated linear units) as used in PaLM
    - LayerScale with a default value of 0.1 (from Mistral/CaIT)

References:
    - https://github.com/facebookresearch/mae
    - https://github.com/lucidrains/x-transformers
"""
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from einops import rearrange, repeat

from model.utils.optimization import get_lr_update
from model.utils.transformer import Block, PatchEmbed, RMSNorm, get_2D_position_embeddings, get_1D_position_embeddings
from model.vitac.tac_model.tactile_encoder import TacConNet
# from voltron.models.vitac.tac_model import instantiate_tac
# Suppress Transformers Logging
transformers.logging.set_verbosity_error()


class VTT_ReAll(nn.Module):
    def __init__(
        self,
        encoder_decoder_cfg: dict,
        train_cfg: dict,
        input_cfg: dict,
        dataset_cfg: dict
    ) -> None:
        """
        Initialize a VCond model with the requisite architecture parameters.

        :param resolution: Base image resolution -- usually 224 (ImageNet size).
        :param patch_size: Height/Width of each patch in pixels -- usually 16.
        :param encoder_depth: Number of Transformer blocks in the encoder -- should be greater than decoder.
        :param encoder_embed_dim: Core embedding/hidden dimension for encoder vision transformer backbone.
        :param encoder_n_heads: Number of heads for encoder multi-headed self-attention.
        :param decoder_depth: Number of Transformer blocks in the decoder -- should be relatively shallow.
        :param decoder_embed_dim: Core embedding/hidden dimension for encoder vision transformer backbone.
        :param decoder_n_heads: Number of heads for encoder multi-headed self-attention.
        :param language_model: Language model to freeze for encoding narrations/utterances.
        :param hf_cache: Cache directory to store pretrained models, for safe distributed training.
        :param language_dim: Dimensionality of the language embedding coming out of the pretrained LM.
        :param tactile_dim: Dimensionality of tactile sensors
        :param optimizer: String denoting which optimizer to use (for MAEs, usually `adamw`)
        :param schedule: Learning rate schedule to use; for Transformers a linear warmup + decay is recommended!
        :param base_lr: Base learning rate, to be scaled via a linear scaling rule (from scaling laws).
        :param min_lr: Minimum learning rate to decay to over the course of learning (usually 0.0)
        :param effective_bsz: Global batch size for update, dictates the scaling of the base_lr.
        :param betas: Adam optimizer betas (only applicable for `adam` and `adamw`. Prevents early loss spiking.
        :param weight_decay: Weight decay for global weight regularization (only applied to non-bias, non-LN layers).
        :param warmup_epochs: Number of epochs to warmup learning rate for linear warmup schedule.
        :param max_epochs: Total number of training epochs to be run.
        :param mask_ratio: Ratio for number of patches to mask out for MAE -- should be fairly high!
        :param mlp_ratio: Ratio for embedding size to Position-wise Feed-Forward MLP (gets shrunk back down).
        :param in_channels: Default number of channels in the base image -- almost always 3.
        :param norm_pixel_loss: Normalize decoder pixel targets for reconstruction (better perf, not interpretable).
        :param use_cls_token: Add <CLS> token for continued pretraining (NOTE: not used in MAE pretraining/finetuning!)
        """
        super().__init__()
        assert list(encoder_decoder_cfg.keys()) == \
               ["encoder_depth", "encoder_embed_dim", "encoder_n_heads",
                "decoder_depth", "decoder_embed_dim", "decoder_n_heads"] and \
               list(train_cfg.keys()) == \
               ["optimizer", "schedule", "base_lr", "min_lr", "effective_bsz",
                "betas", "weight_decay", "mlp_ratio", "norm_pixel_loss", "use_cls_token"] and \
               list(input_cfg.keys()) == \
               ["patch_size", "mask_ratio",
                "tactile_dim", "tactile_en_dim",
                "tactile_type", "tactile_emb_type", "tactile_mask_ratio",
                "tactile_set_to_ZERO", "img_weight", "tac_weight"] and \
               list(dataset_cfg.keys()) == \
               ["resolution", "in_channels", "warmup_epochs", "max_epochs"], \
            "model input error!"
        # Encoder/Decoder Parameters
        (
            self.encoder_depth, self.encoder_embed_dim, self.encoder_n_heads,
            self.decoder_depth, self.decoder_embed_dim, self.decoder_n_heads
        ) = encoder_decoder_cfg.values()
        # General Parameters (for downstream adaptation)
        self.embed_dim, self.n_heads = self.encoder_embed_dim, self.encoder_n_heads
        # Train Parameters
        (
            self.optimizer, self.schedule, self.base_lr, self.min_lr, self.effective_bsz,
            self.betas, self.weight_decay, self.mlp_ratio, self.norm_pixel_loss,
            self.use_cls_token
        ) = train_cfg.values()
        # Input network parameters
        (
            self.patch_size, self.mask_ratio,
            self.tactile_dim, self.tactile_en_dim,
            self.tactile_type, self.tactile_emb_type, self.tac_mask_ratio,
            self.tactile_set_to_ZERO, self.img_weight, self.tac_weight
        ) = input_cfg.values()
        # dataset_parameters
        (
            self.resolution, self.in_channels,
            self.warmup_epochs, self.max_epochs
        ) = dataset_cfg.values()
        self.lr = None

        self.TacConNet = TacConNet(self.tactile_emb_type, self.tactile_dim)

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
            print("Using CLS token!!")

        # MAE Encoder Parameters
        self.patch2embed = PatchEmbed(
            self.resolution, self.patch_size, self.encoder_embed_dim, in_channels=self.in_channels
        )
        self.encoder_pe = nn.Parameter(
            torch.zeros(1, self.patch2embed.num_patches + (1 if self.use_cls_token else 0), self.encoder_embed_dim),
            requires_grad=False,
        )
        self.encoder_pe_tac = self.TacConNet.Encoder_pe(self.encoder_embed_dim, False)

        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    self.encoder_embed_dim,
                    self.encoder_n_heads,
                    self.mlp_ratio,
                    do_rms_norm=True,
                    do_swish_glu=True,
                    do_layer_scale=True,
                )
                for _ in range(self.encoder_depth)
            ]
        )
        self.encoder_norm = RMSNorm(self.encoder_embed_dim)
        # define tactile encoder
        self.encoder_tactile = self.TacConNet.Encoder(self.tactile_dim, self.tactile_en_dim, hidden_size=[512, 512])

        # self.encode_tactile = instantiate_tac(self.tactile_dim, self.tactile_en_dim, hidden_size=[512, 512], model_name=tactile_model)

        # Projection from tactile sensors to Encoder
        self.tactile2encoder = nn.Linear(self.tactile_en_dim, self.encoder_embed_dim)

        # Projection from Encoder to Decoder
        self.encoder2decoder = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim)
        self.encoder2decoder_tac = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim)

        # MAE Decoder Parameters -- Remember the CLS Token (if specified)!
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_pe = nn.Parameter(
            torch.zeros(1, self.patch2embed.num_patches + (1 if self.use_cls_token else 0), self.decoder_embed_dim),
            requires_grad=False,
        )
        self.decoder_pe_tac = self.TacConNet.Decoder_pe(self.decoder_embed_dim, False)
        # 这边不需要cls token了
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    self.decoder_embed_dim,
                    self.decoder_n_heads,
                    self.mlp_ratio,
                    do_rms_norm=True,
                    do_swish_glu=True,
                    do_layer_scale=True,
                )
                for _ in range(self.decoder_depth)
            ]
        )
        self.decoder_norm = RMSNorm(self.decoder_embed_dim)
        self.decoder_prediction = nn.Linear(self.decoder_embed_dim, (self.patch_size**2) * self.in_channels, bias=True)  # (192, )
        self.decoder_prediction_tactile = self.TacConNet.Decoder(self.decoder_embed_dim, self.tactile_dim)
        # self.decoder_prediction_tactile = nn.Linear(self.decoder_embed_dim, 20, bias=True)  # (192, )
        # VCond -- Add "Image" and "Language" Modifier Tokens...
        self.img_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        self.tactile_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))

        # Initialize all Weights
        self.initialize_weights()


    def initialize_weights(self) -> None:
        # Position Encoding -- Fixed 2D Sine-Cosine Embeddings
        enc_pe = get_2D_position_embeddings(
            self.encoder_embed_dim, int(self.patch2embed.num_patches**0.5), cls_token=self.use_cls_token
        )
        self.encoder_pe.data.copy_(torch.from_numpy(enc_pe).float().unsqueeze(0))
        enc_pe_tac = self.TacConNet.Initial_Encoder_pe()
        self.encoder_pe_tac.copy_(torch.from_numpy(enc_pe_tac).float().unsqueeze(0))
        dec_pe = get_2D_position_embeddings(
            self.decoder_embed_dim, int(self.patch2embed.num_patches**0.5), cls_token=self.use_cls_token
        )
        self.decoder_pe.data.copy_(torch.from_numpy(dec_pe).float().unsqueeze(0))
        dec_pe_tac = self.TacConNet.Initial_Decoder_pe()
        self.decoder_pe_tac.copy_(torch.from_numpy(dec_pe_tac).float().unsqueeze(0))
        # Initialize PatchEmbedding as a Linear...
        nn.init.xavier_uniform_(self.patch2embed.proj.weight.data.view([self.patch2embed.proj.weight.data.shape[0], -1]))

        # Initialize Mask Token, Img Token, Lang Token w/ Truncated Normal
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.img_token, std=0.02)
        nn.init.normal_(self.tactile_token, std=0.02)
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)

        # Default Transformer initializer on everything else...
        self.apply(self.transformer_initializer)

    @staticmethod
    def transformer_initializer(m: nn.Module) -> None:
        # Use `xavier_uniform` following Jax ViT
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def encode_language(self, lang: torch.Tensor, lang_mask: torch.Tensor) -> torch.Tensor:
        """Encode language by feeding the *pre-tokenized text* through the frozen language model."""
        self.lm.eval()
        with torch.no_grad():
            transformer_embeddings = self.lm(lang, attention_mask=lang_mask).last_hidden_state
        return transformer_embeddings

    def mask(
        self, patches: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform per-sample random masking by shuffling :: uses argsort random noise to identify masked patches."""
        bsz, n_patches, embed_dim = patches.shape
        if mask_ratio is not None:
            n_keep = int(n_patches * (1 - mask_ratio))
        else:
            n_keep = int(n_patches * (1 - self.mask_ratio))

        # Sample noise of n_patches size, argsort to get shuffled IDs, argsort again to get "unshuffle"
        #   > For clarity -- argsort is an invertible transformation (if argsort `restore`, recovers `shuffle`)
        shuffle_idxs = torch.argsort(torch.rand(bsz, n_patches, device=patches.device), dim=1)
        restore_idxs = torch.argsort(shuffle_idxs, dim=1)

        # Get "keep" (visible) patches
        visible_patches = torch.gather(patches, dim=1, index=shuffle_idxs[:, :n_keep, None].repeat(1, 1, embed_dim))

        # Generate the binary mask --> IMPORTANT :: `0` is keep, `1` is remove (following MAE convention)
        mask = torch.ones(bsz, n_patches, device=patches.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=restore_idxs)

        return visible_patches, mask, restore_idxs

    def get_representations(
        self, img: torch.Tensor, tac: torch.Tensor, mode: str = "patch"
    ) -> torch.Tensor:
        """
        Given either a singleton (img, language) pair or a batch of images and language, extract representations
        subject to the specified mode in < multimodal | visual >.

        :param img: Processed batch of images :: [bsz, 3, 224, 224]
        :param language: Input language as `List[str] | Tuple[str] | None`
        :param mode: Type of representations to extract -- `multimodal` (both vision+text), `visual` (visual only)

        :return: Extracted representations given (img, language) input as sequence.
        """
        assert img.ndim == 4, "Invalid input to `get_representations()`"
        assert mode in {"patch", "cls"}, f"Extraction mode `{mode}` not supported!"
        # Extract desired representations...
        representations = self.encode(img, tac)
        if not self.use_cls_token:
            # return representations[:, :-tac.shape[1], :]
            return representations
        else:
            return representations[:, 1:] if mode == "patch" else representations[:, 0]

    def encode(self, img: torch.Tensor, tac: torch.Tensor) -> torch.Tensor:
        """Default representation extraction function, given a batch of images and tokenized language."""
        touch_embeddings = self.encoder_tactile(tac)
        projected_tac = self.tactile2encoder(touch_embeddings)

        # Patchify
        patches = self.patch2embed(img)

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
            patches = torch.cat([cls_tokens, patches], dim=1)

        # Position Encoding
        patches_pe = patches + self.encoder_pe
        tac_patches_pe = projected_tac + self.encoder_pe_tac

        # Add "modality" embeddings to patches & language
        img_embeddings= patches_pe + self.img_token
        tac_patches = tac_patches_pe + self.tactile_token

        # Create "dummy" visible mask, concatenate image patches & language, feed to Transformer
        patches_mask = torch.ones_like(img_embeddings[..., -1], dtype=torch.float32)
        tactile_mask = torch.ones_like(tac_patches[..., -1], dtype=torch.float32)
        multimodal_embeddings = torch.cat([img_embeddings, tac_patches], dim=1)  # Merge on sequence length...
        multimodal_mask = torch.cat([patches_mask, tactile_mask], dim=1)  # Merge on sequence length...

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            multimodal_embeddings = block(multimodal_embeddings, multimodal_mask)
        multimodal_embeddings = self.encoder_norm(multimodal_embeddings)

        # Return the full sequence of multimodal embeddings...
        return multimodal_embeddings

    def forward_encoder(
        self, img: torch.Tensor, tactile: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # lang_embeddings = self.encode_language(lang, lang_mask)
        # projected_lang = self.lang2encoder(lang_embeddings)

        touch_embeddings = self.encoder_tactile(tactile)
        projected_tac = self.tactile2encoder(touch_embeddings)
        tac_patches_pe = projected_tac + self.encoder_pe_tac

        # Patchify + Position Embedding (without <CLS> Token!)
        patches = self.patch2embed(img)  # (bts, 3, 224, 224) -> (bts, 196, 384)
        patches_pe = patches + (self.encoder_pe if not self.use_cls_token else self.encoder_pe[:, 1:, :])  # (bts, 196, 384) + (1, 196, 384) -> (bts, 196, 384)

        # Create mask (and go ahead and mask out patches at the same time)
        # assert int(self.TacConNet.tac_num_patches * (1- self.tac_mask_ratio)) > 0, \
        #     "tactile mask ratio is too large, visible tac patches is 0 Now!"
        visible_patches, mask, restore_idxs = self.mask(patches_pe, mask_ratio)  # (bts, 49, 384), (bts, 196), (bts, 196)
        tac_visible_patches, tac_mask, tac_restore_idxs = self.mask(tac_patches_pe, self.tac_mask_ratio)

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            cls_token_pe = self.cls_token + self.encoder_pe[:, :1, :]
            cls_tokens = cls_token_pe.expand(img.shape[0], -1, -1)
            visible_patches = torch.cat([cls_tokens, visible_patches], dim=1)

        # Add "modality" embeddings to patches & language
        # visible_patches, projected_lang = visible_patches + self.img_token, projected_lang + self.lang_token
        visible_patches = visible_patches + self.img_token
        tac_patches = tac_visible_patches + self.tactile_token
        # print(f"visible_patches:{visible_patches}")

        # Create "dummy" visible mask, concatenate image patches & language, feed to Transformer
        visible_mask = torch.ones_like(visible_patches[..., -1], dtype=torch.float32)  # (bts, 49)
        tac_visible_mask = torch.ones_like(tac_patches[..., -1], dtype=torch.float32)  # (bts, 2)
        # (bts, 71, 384)
        multimodal_embedding = torch.cat([visible_patches, tac_patches], dim=1)  # Merge on sequence length...
        # (bts, 71)
        multimodal_mask = torch.cat([visible_mask, tac_visible_mask], dim=1)  # Merge on sequence length...
        # print(f"multimodal_embedding:{multimodal_embedding}")
        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            multimodal_embedding = block(multimodal_embedding, multimodal_mask)
        multimodal_embedding = self.encoder_norm(multimodal_embedding)
        # Split multimodal embedding, remove language and return only the visible patches (+ optional <CLS> token)!
        visible_patches = multimodal_embedding[:, : - tac_visible_mask.shape[-1], ...]
        visible_tactile = multimodal_embedding[:, - tac_visible_mask.shape[-1]:, ...]

        return (
            [visible_patches, mask, restore_idxs],
            [visible_tactile, tac_mask, tac_restore_idxs]
        )

    def forward_tac_decoder(self, tactile_patches: torch.Tensor) -> torch.Tensor:
        # project patches into decoder embadding dimension
        projected_tactile = self.encoder2decoder_tac(tactile_patches)
        # print(f"tactile_patches type: {tactile_patches.size()}")
        # print(f"projected_tactile type: {projected_tactile.size()}")

        # Add Position Embeddings
        decoder_patches = projected_tactile + self.decoder_pe_tac   # (bts, 196, 192) + (1, 196, 192) -> (bts, 196, 192)
        # print(f"decoder_patches type: {decoder_patches.size()}")

        # Apply Transformer Blocks...
        for block in self.decoder_blocks:
            decoder_patches = block(decoder_patches)
        decoder_patches = self.decoder_norm(decoder_patches)
        # Run final projection & return --> note <CLS> token handling!
        # print(f"decoder_patches 2 type: {decoder_patches.size()}")
        decoder_prediction = self.decoder_prediction_tactile(decoder_patches)
        # print(f"decoder_prediction 2 type: {decoder_prediction.size()}")
        # return decoder_prediction if not self.use_cls_token else decoder_prediction[:, 1:, :]
        return decoder_prediction

    def forward_decoder(
            self,
            visible_patches: torch.Tensor, restore_idxs: torch.Tensor,
            visible_tactile: torch.Tensor, tac_restore_idxs: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Project patches into decoder embedding dimension
        projected_patches = self.encoder2decoder(visible_patches)# (bts, 49, 384) -> (bts, 49, 192)
        projected_tac = self.encoder2decoder_tac(visible_tactile)

        # print(f"projected_patches.shape:{projected_patches.shape}")
        # print(f"projected_tac.shape:{projected_tac.shape}")

        # Add Mask Tokens to Sequence & Unshuffle
        mask_tokens = self.mask_token.repeat(  # (bts, 147, 192)
            projected_patches.shape[0],
            restore_idxs.shape[1] - visible_patches.shape[1] + (1 if self.use_cls_token else 0),
            1,
        )
        tac_mask_tokens = self.mask_token.repeat(  # (bts, 147, 192)
            projected_tac.shape[0],
            tac_restore_idxs.shape[1] - visible_tactile.shape[1],
            1,
        )

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            # Remove & add back CLS Token as part of the "unshuffling"
            concatenated_patches = torch.cat([projected_patches[:, 1:, :], mask_tokens], dim=1)  # Skip CLS Token
            no_cls_unshuffled_patches = torch.gather(
                concatenated_patches, dim=1, index=restore_idxs[..., None].repeat(1, 1, self.decoder_embed_dim)
            )
            unshuffled_patches = torch.cat([projected_patches[:, :1, :], no_cls_unshuffled_patches], dim=1)
        else:
            concatenated_patches = torch.cat([projected_patches, mask_tokens], dim=1) # (bts, 196, 192)
            unshuffled_patches = torch.gather(     # (bts, 196, 192)
                concatenated_patches, dim=1, index=restore_idxs[..., None].repeat(1, 1, self.decoder_embed_dim)
            )
        # tac patches unshuffle
        tac_concatenated_patches = torch.cat([projected_tac, tac_mask_tokens], dim=1)  # (bts, 196, 192)
        tac_unshuffled_patches = torch.gather(  # (bts, 196, 192)
            tac_concatenated_patches, dim=1, index=tac_restore_idxs[..., None].repeat(1, 1, self.decoder_embed_dim)
        )

        # Add Position Embeddings
        # print(f"unshuffled_patches.shape:{unshuffled_patches.shape}")
        # print(f"self.decoder_pe.shape:{self.decoder_pe.shape}")
        # print(f"self.decoder_pe_tactile.shape:{self.decoder_pe_tactile.shape}")
        assert tac_unshuffled_patches.shape[1:] == self.decoder_pe_tac.shape[1:],\
               "projected_tac.shape != decoder_pe_tactile.shape!"
        decoder_patches = unshuffled_patches + self.decoder_pe   # (bts, 196, 192) + (1, 196, 192) -> (bts, 196, 192)
        decoder_tac = tac_unshuffled_patches + self.decoder_pe_tac

        # print(f"decoder_patches.shape:{decoder_patches.shape}")
        # print(f"decoder_tac.shape:{decoder_tac.shape}")

        # concate decoder patches
        decoder_patches = torch.cat([decoder_patches, decoder_tac], dim=1)
        # print(f"decoder_patches.shape:{decoder_patches.shape}")

        # Apply Transformer Blocks...
        for block in self.decoder_blocks:
            decoder_patches = block(decoder_patches)
        decoder_patches = self.decoder_norm(decoder_patches)    # (bts, 196, 192)
        # print(f"decoder_patches.shape:{decoder_patches.shape}")

        decoder_img_patches = decoder_patches[:, : - self.TacConNet.tac_num_patches, ...]
        decoder_tac_patches = decoder_patches[:, - self.TacConNet.tac_num_patches:, ...]

        # Run final projection & return --> note <CLS> token handling!
        decoder_prediction_img = self.decoder_prediction(decoder_img_patches)  # (bts, 196, 768)
        decoder_prediction_tac = self.decoder_prediction_tactile(decoder_tac_patches)
        # print(f"decoder_prediction:{decoder_prediction.shape}")
        return (
            decoder_prediction_img if not self.use_cls_token else decoder_prediction_img[:, 1:, :],
            decoder_prediction_tac
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert a batch of images to their patched equivalents, by naive reshaping"""
        return rearrange(
            imgs,
            "bsz c (height patch_h) (width patch_w) -> bsz (height width) (patch_h patch_w c)",
            patch_h=self.patch_size,
            patch_w=self.patch_size,
        )  # (bts, 3, 224, 224) -> (bts, 196, 768)

    def depatchify(self, patched_imgs: torch.Tensor) -> torch.Tensor:
        """Convert  patched imgs to their origin batch of images, by naive reshaping"""
        return rearrange(
            patched_imgs,
            "bsz (height width) (patch_h patch_w c) -> bsz c (height patch_h) (width patch_w)",
            patch_h=self.patch_size,
            patch_w=self.patch_size,
            height=int(self.resolution**0.5),
            width=int(self.resolution**0.5),
            c=self.in_channels
        )

    def generate_origin_img(self, patched_imgs, ori_imgs):
        targets = self.patchify(ori_imgs)
        # Normalize targets parameters
        mu, var = targets.mean(dim=-1, keepdim=True), targets.var(dim=-1, unbiased=True, keepdim=True)
        # targets = (targets - mu) / ((var + 1e-6) ** 0.5)

        # de normalize patched_imgs
        recon_patches = patched_imgs * ((var + 1e-6) ** 0.5) + mu

        # depatchify
        recon_imgs = self.depatchify(recon_patches)

        return recon_imgs

    def compute_tac_loss(self, tactile: torch.Tensor, reconstructions: torch.Tensor):
        # # Normalize targets
        # mu, var = tactile.mean(keepdim=True), tactile.var(unbiased=True, keepdim=True)
        # targets = (tactile - mu) / ((var + 1e-6) ** 0.5)

        if self.tactile_type in {"binary"}:
            # Loss = torch.nn.BCEWithLogitsLoss()
            # tactile_loss = Loss(reconstructions, tactile)
            mse = (reconstructions - tactile) ** 2
            avg_loss_per_patch = mse.mean(dim=-1)
            tactile_loss = avg_loss_per_patch.sum()
        # compute mean loss
        elif self.tactile_type in {"normalize"}:
            mse = (reconstructions - tactile) ** 2
            avg_loss_per_patch = mse.mean(dim=-1)
            tactile_loss = avg_loss_per_patch.sum()
        else:
            raise KeyError(f"compute loss for tactile type \"{self.tactile_type}\" is not implemented!")

        return tactile_loss

    def compute_loss(self, imgs: torch.Tensor, reconstructions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert self.norm_pixel_loss, "`norm_pixel_loss` should always be true... false only for visualizations!"
        targets = self.patchify(imgs)

        # Normalize targets...
        mu, var = targets.mean(dim=-1, keepdim=True), targets.var(dim=-1, unbiased=True, keepdim=True)
        targets = (targets - mu) / ((var + 1e-6) ** 0.5)

        # Compute mean loss per patch first...
        mse = (reconstructions - targets) ** 2
        avg_loss_per_patch = mse.mean(dim=-1)

        # # Compute mean loss only on *removed* patches and return
        # return (avg_loss_per_patch * mask).sum() / mask.sum()
        # Compute mean loss on ALL patches
        return avg_loss_per_patch.sum()

    def forward(
        self, img: torch.Tensor, tactile: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        # tactile input process
        tactile_input = tactile * int(not self.tactile_set_to_ZERO)
        # forward
        ([visible_patches, mask, restore_idxs],
         [visible_tactile, tactile_mask, tac_restore_idxs]) = self.forward_encoder(img, tactile_input, mask_ratio)
        (img_reconstructions,
         tac_reconstructions) = self.forward_decoder(visible_patches, restore_idxs, visible_tactile, tac_restore_idxs)

        # tac_reconstructions = self.forward_tac_decoder(visible_tactile)
        # compute union loss
        img_loss = self.compute_loss(img, img_reconstructions, mask)# (bts, 3, 224, 224) , (bts, 196, 768), (bts, 196)
        tac_loss = self.compute_tac_loss(torch.Tensor(tactile), tac_reconstructions)
        loss = self.img_weight * img_loss + self.tac_weight * tac_loss

        return loss, [img_reconstructions, tac_reconstructions], [img_loss, tac_loss], mask

    def configure_optimizer(self) -> Tuple[torch.optim.Optimizer, Callable[[int, float], float]]:
        # Short-Circuit on Valid Optimizers
        if self.optimizer not in ["adamw"]:
            raise NotImplementedError(f"Optimizer `{self.optimizer}` not supported - try [`adamw`] instead!")

        # Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed...
        #   > This is a compact rewrite of `param_groups_weight_decay()` from TIMM because I don't want the dependency
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Check on any parameters with fewer than 2 dimensions or with "bias" in the name...
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        # Build Parameter Groups
        groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

        # Compute LR -- MAE uses the `linear scaling rule` :: lr = base_lr * (effective_bsz / 256)
        #   > https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md
        self.lr = self.base_lr * (self.effective_bsz / 256)

        # Create Optimizer & LR Scheduler
        optimizer = torch.optim.AdamW(groups, lr=self.lr, betas=self.betas)
        update_lr = get_lr_update(optimizer, self.schedule, self.lr, self.min_lr, self.warmup_epochs, self.max_epochs)
        return optimizer, update_lr
