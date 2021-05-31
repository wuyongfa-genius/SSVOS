"""Simple implementation of Bootstrap Your Own Latent (BYOL).
"""
import torch
from torch import nn
from torch.nn import functional as F
import copy
from functools import wraps
from  ssvos.models.backbones.torchvision_resnet import resnet50, resnet18

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def loss_fn(x, y):
    x = torch.stack(x, dim=1) # B,N,C
    y = torch.stack(y, dim=1) # B,N,C
    if y.requires_grad:
        y.detach_()
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    cosine_similarity = torch.einsum('bic, bjc -> bij', x, y)
    # return 2 - 2 * (x * y).sum(dim=-1)
    return 2-2*cosine_similarity

class EMA(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def _update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def forward(self, target_encoder, online_encoder):
        for online_params, target_params in zip(online_encoder.parameters(), target_encoder.parameters()):
            old_weight, up_weight = target_params.data, online_params.data
            target_params.data = self._update_average(old_weight, up_weight)


class MLP(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=4096, out_dim=256):
        super().__init__()
        if isinstance(hidden_dim, int):
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim), # should set bias to False, cause it's followed by BN
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        elif isinstance(hidden_dim, (list, tuple)):  # two hidden layers
            assert len(hidden_dim) == 2
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim[0]), # should set bias to False, cause it's followed by BN
                nn.BatchNorm1d(hidden_dim[0]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim[0], hidden_dim[1]), # should set bias to False, cause it's followed by BN
                nn.BatchNorm1d(hidden_dim[1]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim[1], out_dim), # should set bias to False, cause it's followed by BN
                nn.BatchNorm1d(out_dim, affine=False)
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm1d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class NetWrapper(nn.Module):
    """"Wrap an encoder with a projection MLP in a single Module."""

    def __init__(self, encoder: nn.Module, mlp:nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projector = mlp

    def forward(self, x):
        representation = self.encoder(x)
        projection = self.projector(representation)
        return projection


class BYOL(nn.Module):
    def __init__(self, 
            encoder: nn.Module, 
            feat_dim=2048, 
            projector_hidden_dim=[2048,2048], 
            out_dim=2048, 
            predictor_hidden_dim=512,
            moving_average_decay=None,
            simsiam=True,
            ):
        super().__init__()
        ## whether to use simsiam
        self.simsiam = simsiam
        if predictor_hidden_dim is None:
            assert simsiam==False
            beta = moving_average_decay or 0.99
            ## ema_update
            self.ema_updater = EMA(beta)
        if simsiam:
            assert moving_average_decay is None
        ## encoder with projector
        self.projector_mlp = MLP(feat_dim, projector_hidden_dim, out_dim)
        if hasattr(encoder, 'fc'):
            encoder.fc = nn.Identity()
        if hasattr(encoder, 'head'):
            encoder.head = nn.Identity()
        self.online_encoder = NetWrapper(encoder, self.projector_mlp)
        self.target_encoder = None
        ## predictor
        hidden_dims = predictor_hidden_dim or projector_hidden_dim
        self.predictor = MLP(out_dim, hidden_dims, out_dim)

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        self.ema_updater(self.target_encoder, self.online_encoder)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        def set_requires_grad(model, val):
            for p in model.parameters():
                p.requires_grad = val
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder
    
    def forward(self, frames1:list, frames2:list):
        ## forward online encoder projector and predictor
        online_predictions1 = []
        online_predictions2 = []
        for f1, f2 in zip(frames1, frames2):
            online_predictions1.append(self.predictor(self.online_encoder(f1)))
            online_predictions2.append(self.predictor(self.online_encoder(f2)))
        ## forward target encoder
        with torch.no_grad():
            target_encoder = self._get_target_encoder() if not self.simsiam else self.online_encoder
            target_projections1 = []
            target_projections2 = []
            for f1, f2 in zip(frames1, frames2):
                tar_p1 = target_encoder(f1)
                tar_p1.detach_()
                tar_p2 = target_encoder(f2)
                tar_p2.detach_()
                target_projections1.append(tar_p1)
                target_projections2.append(tar_p2)
        ## compute loss
        loss = 1/2*loss_fn(online_predictions1, target_projections2)
        loss += 1/2*loss_fn(online_predictions2, target_projections1)

        return loss.mean()
        

# if __name__=="__main__":
#     # byol = BYOL(encoder=resnet50(), simsiam=False).cuda()
#     simsiam = BYOL(resnet18(zero_init_residual=True), feat_dim=512, projector_hidden_dim=[2048,2048], out_dim=2048, predictor_hidden_dim=512, simsiam=True).cuda()
#     frames1 = []
#     frames2 = []
#     for i in range(4):
#         frames1.append(torch.randn(2, 3, 224, 224).cuda())
#         frames2.append(torch.randn(2, 3, 224, 224).cuda())
#     loss = simsiam(frames1, frames2)
#     print(loss)
