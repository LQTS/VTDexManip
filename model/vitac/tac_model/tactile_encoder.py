import torch
import torch.nn as nn
from typing import Optional, List
from model.utils.transformer import get_1D_position_embeddings
from model.vitac.pygcn.layers import GraphConvolution
from model.vitac.pygcn.graph import Graph
from torch.nn import Parameter
class TacConNet:
    def __init__(self, embed_type, touch_dim):
        self.embed_type = embed_type
        self.touch_dim = touch_dim
        self.add_pe = True
        if self.embed_type == "flatten":
            self.tac_num_patches = 1
        elif self.embed_type == "tips_palm":
            self.tac_num_patches = 2
        elif self.embed_type == "5fingers":
            self.tac_num_patches = 5
        elif self.embed_type == "separate":
            self.tac_num_patches = 20
        elif self.embed_type == "separate_nope":
            self.add_pe = False
            self.tac_num_patches = 20
        elif self.embed_type == "gcn":
            self.tac_num_patches = 20
        else:
            raise KeyError(f"TacConNet.embed_type \"{self.embed_type}\" is not implemented")

        print(f"TacConNet initialize successfully! Add Pe: {float(self.add_pe)}")

    def Encoder(self,
                touch_dim: int,
                embed_dim: int,
                hidden_size: Optional[List[int]] = None
                ) -> nn.Module:
        if self.embed_type == "flatten":
            assert hidden_size is not None and len(hidden_size) == 2, \
                  f"wrong hidden_size for embed_type {self.embed_type}!"
            return TacMLP1D_Enc(touch_dim, embed_dim, hidden_size)
        elif self.embed_type == "tips_palm":
            assert hidden_size is not None and len(hidden_size) == 2, \
                  f"wrong hidden_size for embed_type {self.embed_type}!"
            return TacMLP2D_Enc(touch_dim, embed_dim, hidden_size)
        elif self.embed_type == "5fingers":
            return TacFC5_Enc(touch_dim, embed_dim)
        elif self.embed_type in {"separate", "separate_nope"}:
            return TacFC20_Enc(touch_dim, embed_dim)
        elif self.embed_type == "gcn":
            return TacGCN20_Enc(touch_dim, embed_dim)

    def Decoder(self,
                embed_dim: int,
                touch_dim: int,
                ) -> nn.Module:
        if self.embed_type == "flatten":
            return TacMLP1D_Dec(embed_dim, touch_dim)
        elif self.embed_type == "tips_palm":
            return TacMLP2D_Dec(embed_dim, touch_dim)
        elif self.embed_type == "5fingers":
            return TacFC5_Dec(embed_dim, touch_dim)
        elif self.embed_type in {"separate", "separate_nope"}:
            return TacFC20_Dec(embed_dim, touch_dim)
        elif self.embed_type == "gcn":
            return TacGCN20_Dec(embed_dim, touch_dim)
    def Encoder_pe(self, encoder_embed_dim, use_cls):
        self.encoder_embed_dim = encoder_embed_dim
        return nn.Parameter(
            torch.zeros(1, self.tac_num_patches + (1 if use_cls else 0), encoder_embed_dim),
            requires_grad=False,
        )

    def Decoder_pe(self, decoder_embed_dim, use_cls):
        self.decoder_embed_dim = decoder_embed_dim
        return nn.Parameter(
            torch.zeros(1, self.tac_num_patches + (1 if use_cls else 0), decoder_embed_dim),
            requires_grad=False,
        )

    def Initial_Encoder_pe(self):
        return get_1D_position_embeddings(
            self.encoder_embed_dim, self.tac_num_patches + 1
        )[1:] * float(self.add_pe)

    def Initial_Decoder_pe(self):
        return get_1D_position_embeddings(
            self.decoder_embed_dim, self.tac_num_patches + 1
        )[1:] * float(self.add_pe)



# Tactile Embedding Module (MLP)
class TacMLP1D_Enc(nn.Module):
    def __init__(
        self,
        touch_dim: int,
        embed_dim: int,
        hidden_size: list
    ):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(touch_dim, hidden_size[0]),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size[0], hidden_size[1]),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size[1], embed_dim))

    def forward(self, touch_sensors: torch.Tensor) -> torch.Tensor:
        # (x,20) -> (x, 1, 20)
        touch_sensors = touch_sensors.unsqueeze(-2)
        tac_embeddings = self.encoder(touch_sensors)

        return tac_embeddings

class TacMLP1D_Dec(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        touch_dim: int
    ):
        super().__init__()
        self.decoder = nn.Linear(embed_dim, touch_dim, bias=True)

    def forward(self, touch_output: torch.Tensor) -> torch.Tensor:
        # (x,20) -> (x, 1, 20)
        tac_recons = self.decoder(touch_output)

        return tac_recons


class TacMLP2D_Enc(nn.Module):
    def __init__(
        self,
        touch_dim: int,
        embed_dim: int,
        hidden_size: list
    ):
        super().__init__()
        assert touch_dim == 20, "tactile dimension is not 20! Please update TacMLP1DV2 model"
        # 指尖5维
        self.encoder_fingertip = nn.Sequential(nn.Linear(5, hidden_size[0]),
                                               nn.ReLU(),
                                               nn.Linear(hidden_size[0], hidden_size[1]),
                                               nn.ReLU(),
                                               nn.Linear(hidden_size[1], embed_dim))
        # 其他15维
        self.encoder_palm = nn.Sequential(nn.Linear(15, hidden_size[0]),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size[0], hidden_size[1]),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size[1], embed_dim))

    def forward(self, touch_sensors: torch.Tensor) -> torch.Tensor:
        # (x,20) -> (x, 1, 20)
        touch_sensors = touch_sensors.unsqueeze(-2)
        touch_fingertips = touch_sensors[:, :, :5]
        touch_palm = touch_sensors[:, :, 5:]
        tac_embeddings_fingertip = self.encoder_fingertip(touch_fingertips)
        tac_embeddings_palm = self.encoder_palm(touch_palm)
        tac_embeddings = torch.cat([tac_embeddings_fingertip, tac_embeddings_palm], dim=1)

        return tac_embeddings

class TacMLP2D_Dec(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        touch_dim: int
    ):
        super().__init__()
        assert touch_dim == 20, "tactile dimension is not 20! Please update TacMLP2D_Dec module"
        # 指尖5维
        self.decoder_fingertip = nn.Linear(embed_dim, 5, bias=True)
        # 其他15维
        self.decoder_palm = nn.Linear(embed_dim, 15, bias=True)

    def forward(self, touch_output: torch.Tensor) -> torch.Tensor:
        # (x,2, 384) -> (x, 1, 20)
        touch_fingertips = self.decoder_fingertip(touch_output[:, 0, :])
        touch_palm = self.decoder_palm(touch_output[:, 1, :])
        tac_recons = torch.cat([touch_fingertips, touch_palm], dim=1)

        return tac_recons

class TacFC5_Enc(nn.Module):
    def __init__(
        self,
        touch_dim: int,
        embed_dim: int
    ):
        super().__init__()
        assert touch_dim == 20, "tactile dimension is not 20! Please update TacFC5_Enc model"
        # 5根手指
        self.fingers = nn.ModuleList()
        for _ in range(5):
            self.fingers.append(nn.Linear(4, embed_dim))


    def forward(self, touch_sensors: torch.Tensor) -> torch.Tensor:
        # (x,20) -> (x, 1, 20)
        touch_sensors = touch_sensors.unsqueeze(-2)
        tac_embeddings_list = [
            self.fingers[i](touch_sensors[:, :, [i, i+5, i+10, i+15]]) for i in range(5)
            #
            # self.fingers[0](touch_sensors[:, :, :4]),
            # self.fingers[1](touch_sensors[:, :, 4:8]),
            # self.fingers[2](touch_sensors[:, :, 8:12]),
            # self.fingers[3](touch_sensors[:, :, 12:16]),
            # self.fingers[4](touch_sensors[:, :, 16:]),
        ]
        tac_embeddings = torch.cat(tac_embeddings_list, dim=1)

        return tac_embeddings


class TacFC5_Dec(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        touch_dim: int,
    ):
        super().__init__()
        assert touch_dim == 20, "tactile dimension is not 20! Please update TacFC5_Dec module"
        # 5根手指
        self.fingers = nn.ModuleList()
        for _ in range(5):
            self.fingers.append(nn.Linear(embed_dim, 4))

    def forward(self, touch_output: torch.Tensor) -> torch.Tensor:
        # (x,20) -> (x, 1, 20)
        tac_embeddings_list = [self.fingers[i](touch_output[:, i, :]) for i in range(5)]
        tac_embeddings = torch.cat(tac_embeddings_list, dim=1)

        return tac_embeddings


class TacFC20_Enc(nn.Module):
    def __init__(
        self,
        touch_dim: int,
        embed_dim: int
    ):
        super().__init__()
        assert touch_dim == 20, "tactile dimension is not 20! Please update TacFC5_Enc model"
        # 20个通道
        self.fingers = nn.ModuleList()
        for _ in range(20):
            self.fingers.append(nn.Linear(1, embed_dim))

    def forward(self, touch_sensors: torch.Tensor) -> torch.Tensor:
        # (x,20) -> (x, 1, 20)
        touch_sensors = touch_sensors.unsqueeze(-2)
        tac_embeddings_list = [
            self.fingers[i](touch_sensors[:, :, i]).unsqueeze(-2) for i in range(20)
        ]
        tac_embeddings = torch.cat(tac_embeddings_list, dim=1)

        return tac_embeddings


class TacFC20_Dec(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        touch_dim: int,
    ):
        super().__init__()
        assert touch_dim == 20, "tactile dimension is not 20! Please update TacFC5_Dec module"
        # 20维
        self.fingers = nn.ModuleList()
        for _ in range(20):
            self.fingers.append(nn.Linear(embed_dim, 1))

    def forward(self, touch_output: torch.Tensor) -> torch.Tensor:
        # (x,20) -> (x, 1, 20)
        tac_embeddings_list = [self.fingers[i](touch_output[:, i, :]) for i in range(20)]
        tac_embeddings = torch.cat(tac_embeddings_list, dim=1)

        return tac_embeddings

class TacGCN20_Enc(nn.Module):
    def __init__(
            self,
            touch_dim: int,
            embed_dim: int,
    ):
        super().__init__()
        assert touch_dim == 20, "tactile dimension is not 20! Please update TacFC5_Dec module"

        self.gc = GraphConvolution(1, embed_dim)
        self.graph = Graph()
        self.adj = Parameter(torch.FloatTensor(self.graph.A), requires_grad=False)

    def forward(self, x):
        B, N = x.size()
        # x = x.view(-1, 1) # (B, N)-> (B*N, 1)
        y = self.gc(x.unsqueeze(-1), self.adj.repeat(B, 1, 1))

        return y

class TacGCN20_Dec(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            touch_dim: int,
    ):
        super().__init__()
        assert touch_dim == 20, "tactile dimension is not 20! Please update TacFC5_Dec module"

        self.gc = GraphConvolution(embed_dim, 1)
        self.graph = Graph()
        self.adj = Parameter(torch.FloatTensor(self.graph.A), requires_grad=False)

    def forward(self, x):
        B, N, C = x.size()
        # x = x.unsqueeze(-1)
        y = self.gc(x, self.adj.repeat(B, 1, 1))

        return y.squeeze()