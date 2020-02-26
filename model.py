import torch
from torch import nn
import torch.nn.functional as F


class NASH(nn.Module) :
    def __init__(self, config, input_size) :
        super(NASH, self).__init__()
        
        self.dropout = nn.Dropout(config.dropout)
        self.sigmoid = nn.Sigmoid()

        # encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_size, config.hidden_size)
            , nn.ReLU()
            , nn.Dropout(config.dropout)
            , nn.Linear(config.hidden_size, config.output_size)
            , nn.Sigmoid()
        )

        # decoder network
        self.decoder = nn.Linear(config.output_size, input_size)

        # noise network
        self.sigma = nn.Linear(config.output_size, config.output_size)

        self.deterministic = config.deterministic
        self.mu = None
        if not self.deterministic :
            self.mu = torch.rand(config.output_size)
            if config.use_cuda :
                self.mu = self.mu.cuda()

    def binarization(self, x) :
        if self.deterministic :
            # deterministic binarization
            mu = 0.5
        else :
            # stochastic binarization
            if self.mu is None :
                self.mu = torch.rand(x.size(-1)).cuda()
            mu = self.mu
        code = (torch.sign(x - mu) + 1) / 2
        return code

    def encode(self, x) :
        z = self.encoder(x)
        z_quantized = self.binarization(z)
        return z_quantized

    def forward(self, x) :
        # get hash code
        z = self.encoder(x)

        # kl term loss
        kl_loss = torch.mean(torch.sum(
            z * torch.log(z / 0.5) + 
            (1-z) * torch.log((1 - z) / 0.5)
            , dim=-1))

        z_quantized = self.binarization(z)

        # Straight-Throught estimation
        # preserve reconstruction loss gradient of quantized latent to continuous latent z
        z_quantized = z + (z_quantized-z).detach()
        z_quantized = self.dropout(z_quantized)

        # add noise to code
        z_noise = z_quantized + self.dropout(self.sigmoid(self.sigma(z)))

        # reconstruct bag-of-words
        logit = self.decoder(z_noise)
        prob = F.log_softmax(logit, dim=-1)
        x_onehot = (~(x==0)).long() # bag-of-words representation
        rec_loss = torch.mean(torch.sum(prob*x_onehot.float(), dim=-1)/torch.sum(x_onehot, dim=-1))

        # sum of loss
        loss = -(rec_loss-kl_loss)

        return loss