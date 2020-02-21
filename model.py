import torch
from torch import nn
import torch.nn.functional as F

class NASH(nn.Module) :
    def __init__(self, input_size
                , hidden_size=500
                , output_size=128
                , layer_num=2
                , dropout=0.1
                , deterministic=True
                ) :
        super(NASH, self).__init__()
        
        # build encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size)
            , nn.ReLU()
            , nn.Dropout(dropout)
            , nn.Linear(hidden_size, output_size)
            , nn.Sigmoid()
            )

        # decoder embedding
        self.E = nn.Embedding(input_size, output_size)
        
        # network for gaussian noise
        self.std = nn.Linear(output_size, output_size)

        self.deterministic = deterministic

    def binarization(self, x) :
        if self.deterministic :
            # deterministic binarization
            mu = 0.5
        else :
            # stochastic binarization
            mu = torch.rand(x.size())
        code = (torch.sign(x - mu) + 1) / 2
        return code, mu

    def forward(self, x) :
        #print(x.size())

        z = self.encoder(x)
        #print(z.size())

        #KL = z*torch.log()

        z_quantized, mu = self.binarization(z)
        #print(z_quantized.size())

        kl_loss = torch.sum(z*torch.log(z/mu) + (1-z)*torch.log((1-z)/(1-mu)))
        #print("kl")
        #print(kl_loss)

        # Straight-Throught estimation
        # preserve reconstruction gradient of quantized latent to continous latent z
        z_quantized = z + (z_quantized-z).detach()
        #print(z_quantized)

        z_noise = z_quantized + self.std(z)
        #print(z_noise)

        x_onehot = (~(x==0)).long()
        #print(x_onehot.size())

        x_hat = self.E(x_onehot)
        #print(x_hat.size())

        logit = torch.matmul(z_noise.unsqueeze(1), x_hat.transpose(-2, -1)).squeeze(1)
        #print(logit.size())

        prob = F.log_softmax(logit, dim=-1)
        #print(prob.size())

        loss = -torch.mean(torch.sum(prob*x_onehot.float(), dim=1))
        #print(loss)

        return loss, kl_loss