"""An implementation of Deep-KSVD"""
import torch
from torch import nn
from torch import linalg as LA
from torch.nn import functional as F

def soft_thresh(x, threshold):
    return torch.sign(x)*F.relu(torch.abs(x)-threshold)

class MLP(nn.Module):
    """A MLP used to predict the soft threshold."""
    def __init__(self, in_dim, out_dim=1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim*2),
            nn.BatchNorm1d(in_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim*2, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim//2),
            nn.BatchNorm1d(in_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, out_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class Learnable_K_SVD(nn.Module):
    def __init__(self, feature_dim, dict_atoms, iters) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.dict_atoms = dict_atoms
        self.iters = iters
        self.lambda_predictor = MLP(feature_dim)
        dictionary, squared_spectral_norm = self._init_dict()
        self.dictionary = nn.Parameter(dictionary) # d*N
        self.c = nn.Parameter(squared_spectral_norm)
        self.e = self.register_buffer('Identity', torch.eye(dictionary.shape[-1]))
    
    def _init_dict(self):
        dictionary = torch.randn((self.feature_dim, self.dict_atoms)) # d*N
        ## normalize
        dictionary = F.normalize(dictionary, dim=1)
        ## compute the spectral norm of dict
        spectral_norm = LA.norm(dictionary, ord=2)
        return dictionary, spectral_norm**2
    
    def forward(self, x):
        lamd = self.lambda_predictor(x)
        thresh = lamd/self.c
        ## iterative soft-threshold shrinkage main iteration
        M1 = self.e-1/self.c*(torch.matmul(self.dictionary.T, self.dictionary)) #N*N
        M2 = 1/self.c*(torch.matmul(self.dictionary.T, x.T)) # N*B
        alpha = soft_thresh(M2, thresh) # N*B
        for _ in range(self.iters):
            alpha = soft_thresh(torch.matmul(M1, alpha)+M2, thresh)
        ## reconstruct x from alpha and dict
        reconstructed_x = torch.matmul(self.dictionary, alpha).T
        return reconstructed_x


if __name__=="__main__":
    X = torch.tensor([1.5, 0.1, -0.1, -1.6])
    print(soft_thresh(X, 1))