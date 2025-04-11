import torch 

class KLD():
    def __init__(self, std = 1):
        self.std = torch.tensor(std)

    # def kld(self, mu, logvar):
    #     kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #     return kl_divergence

    def kld(self, mu, logvar):
        x = self.std
        kl_divergence = 0.5 * torch.sum(torch.log(self.std ** 2) - logvar + ((mu.pow(2) + logvar.exp()) / self.std**2) - 1)
        return kl_divergence