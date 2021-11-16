# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import distributed as dist
from einops import rearrange
from ssvos.models.deep_k_svd import Learnable_K_SVD

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: dist.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * dist.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


class MoCo_ResNet_K_SVD(MoCo):
    def __init__(self, base_encoder, dict_atoms, iters, dim=256, mlp_dim=4096, T=1):
        super().__init__(base_encoder, dim=dim, mlp_dim=mlp_dim, T=T)
        self.dict_atoms = dict_atoms
        self.iters = iters
    
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # set predictor as a deep K-SVD 
        self.predictor = Learnable_K_SVD(dim, self.dict_atoms, self.iters)
    
    # def contrastive_loss(self, q, k):
    #     B,T,C = q.shape
    #     q = rearrange(q, 'b t c ->(b t) c')
    #     k = rearrange(k, 'b t c ->(b t) c')
    #     similarity_matrix = F.cosine_similarity(q, k)
    #     labels = [torch.ones((T,T), dtype=torch.bool).cuda() for _ in range(B)]
    #     label_mask = torch.block_diag(*labels)
    #     # positive pair loss
    #     positive_loss = torch.sum(label_mask*(1-similarity_matrix))
    #     negative_loss = torch.sum((~label_mask)*(1+similarity_matrix))

    #     return (positive_loss+negative_loss) / ((B*T)**2)
    
    def contrastive_loss(self, q, k):
        B,T,C = q.shape
        q = rearrange(q, 'b t c ->(b t) c')
        k = rearrange(k, 'b t c ->(b t) c')
        k = concat_all_gather(k) # (gpus*b*t)*c
        similarity_matrix = F.cosine_similarity(q, k)
        ## make label mask
        world_size = dist.get_world_size()
        label_mask = torch.zeros((world_size*B*T, B*T), dtype=torch.bool).cuda()
        # positive pairs only exist at the same device
        labels = [torch.ones((T,T), dtype=torch.bool).cuda() for _ in range(B)]
        pos_area_label_mask = torch.block_diag(*labels) #(BT)*(BT)
        rank = dist.get_rank()
        label_mask[rank*B*T:(rank+1)*B*T,:] = pos_area_label_mask
        ## compute InfoNCE loss
        logits_matrix = torch.exp(similarity_matrix/self.T)
        pos_logits_sum = torch.sum(logits_matrix*label_mask, dim=1)
        all_logits_sum = torch.sum(logits_matrix, dim=1)

        return -(2*self.T)*torch.log(pos_logits_sum/all_logits_sum).mean()
    
    def forward(self, x, m):
        B,T,C,H,W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        ##compute query features
        q = self.predictor(self.base_encoder(x))
        with torch.no_grad():
            self._update_momentum_encoder(m)  # update the momentum encoder
            k = self.momentum_encoder(x)
        q = rearrange(q, '(b t) c -> b t c', b=B) # B,T,C
        k = rearrange(k, '(b t) c -> b t c', b=B) # B,T,C
        ## seperate q1,q2,k1,k2
        q1, q2 = q[:,:T//2,:], q[:,T//2:,:]
        k1, k2 = k[:,:T//2,:], k[:,T//2:,:]

        return 1/2*(self.contrastive_loss(q1, k2)+self.contrastive_loss(q2, k1))

if __name__=="__main__":
    x = torch.tensor([[1,2],[3,4]])
    print(rearrange(x, 'a b -> (a b)'))
    print(rearrange(x, 'a b ->(b a)'))