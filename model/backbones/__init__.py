from .FCT import build_model
from .unet import MAP_UNet, Sim_UNet

model_dict = {
    "fct": build_model,
    "unet_cat": MAP_UNet,
    "unet_sim": Sim_UNet
}