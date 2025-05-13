from .tactile_encoder import TacMLP1D, TacMLP1DV2

def instantiate_tac(in_dim:int, out_dim:int, hidden_size:list, model_name:str):

    tac_model = eval(model_name)(
        touch_dim=in_dim,
        embed_dim=out_dim,
        hidden_size=hidden_size
    )

    return  tac_model