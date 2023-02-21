import torch
import torchvision
from torch import nn
from copy import deepcopy

class FTT_resnet(nn.Module):
    def __init__(self, 
                 name: str, 
                 sup_fraction: float,
                 unsup_fraction: float,
                 pretrained: bool = True, 
                 n_classes: int = 2) -> None:
        super().__init__()
        assert name in ["resnet18", "resnet50"], ValueError(
            f"model name could only be chosen from `resnet18` and `resnet50`, but is {name}.")
        if name == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=pretrained)
        elif name == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=pretrained)

        self.n_channels = self.model.fc.in_features # n_channels output by CNN part
        self.model_dim = int(sup_fraction * self.n_channels) 
        self.bimodel_dim = int(unsup_fraction * self.n_channels) 
        self.n_features = self.model_dim + self.bimodel_dim

        self.use_sup = bool(self.model_dim > 0) 
        self.use_unsup = bool(self.bimodel_dim > 0)

        # Initialize the bimodel and set requires_grad to False.
        self.bimodel = deepcopy(self.model)
        if self.use_sup:
            self.model.fc = nn.Linear(self.n_channels, self.model_dim)
        else:
            self.model = None
        if self.use_unsup:
            self.bimodel.fc = nn.Linear(self.n_channels, self.bimodel_dim)
            # Do not update parameters for bimodel
            for param in self.bimodel.parameters():
                param.requires_grad = False
            self.bimodel.eval() # make sure that bimodel always lie in the eval mode
        else:
            self.bimodel = None

        self.classifier = nn.Linear(self.n_features, n_classes)

    def get_features(self, x):
        if self.use_sup and self.use_unsup:
            return torch.concat([self.model(x), self.bimodel(x)], dim = -1)
        elif self.use_sup:
            return self.model(x)
        else:
            return self.bimodel(x)

    def train(self):
        if self.use_sup:
            self.model.train()
        if self.use_unsup:
            self.bimodel.eval()

    def eval(self):
        if self.use_sup:
            self.model.eval()
        if self.use_unsup:
            self.bimodel.eval()

    def forward(self, x):
        feature = self.get_features(x)
        return self.classifier(feature)

    def get_embed(self, x, to_numpy = True):
        feature = self.get_features(x)
        return feature if not to_numpy else feature.detach().cpu().numpy()
    

    def set_bimodel_fc(self, weight, bias):
        assert bool(self.use_unsup) is True
        dtype = self.bimodel.fc.weight.dtype
        weight = torch.tensor(weight, dtype = dtype).cuda()
        bias = torch.tensor(bias, dtype = dtype).cuda()
        self.bimodel.fc.weight.set_(weight)
        self.bimodel.fc.bias.set_(bias)

