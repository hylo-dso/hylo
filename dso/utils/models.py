from ..models.cifar import resnet32, mobilenetv2, densenet
from torchvision.models import resnet50
from .cnn_utils.unet import UNet

def get_model(name, **kwargs):
    networks = {
        'densenet': densenet,
        'mobilenetv2': mobilenetv2,
        'resnet32': resnet32,
        'resnet50': resnet50,
        'unet': UNet
    }

    if name not in networks:
        raise NotImplementedError

    return networks[name](**kwargs)
