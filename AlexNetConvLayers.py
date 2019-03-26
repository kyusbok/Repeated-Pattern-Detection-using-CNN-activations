from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch import nn


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNetConvLayers(models.AlexNet):

    def __init__(self):
        super(AlexNetConvLayers, self).__init__()

    def forward(self, x):
        layers_outputs = []
        for l in self.features:
            x = l(x)
            if isinstance(l, nn.Conv2d):
                layers_outputs.append(x)

        return layers_outputs


def alexnet_conv_layers():
    model = AlexNetConvLayers()
    model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
