import torch
from enc_dec import Encoder, Decoder


def gaussian_likelihood_loss(y, mu, logvar):
    # 分散行列Varが対角成分var_iの対角行列の場合には、log(det|Var|) = log(prod_i var_i) = sum_i log(var_i)
    return 0.5 * torch.mean(((y-mu)**2) * torch.exp(-logvar) + logvar)


def kld(mu1, logvar1, mu2, logvar2):
    # kld(p1|p2) = E_{z~p1}[ log p1(z) - log p2(z) ]
    tmp1 = 0.5 * (logvar2 - logvar1) # log (sigma2/sigma1)
    tmp2 = 0.5*(torch.exp(logvar1)+(mu1-mu2)**2) / torch.exp(logvar2) # (sigma1^2+(mu1-mu2)^2)/(2*sigma2^2)
    return torch.mean(tmp1 + tmp2)



class unweightedVI(torch.nn.Module):
    def __init__(self, s_dim, a_dim, z_dim, g_dim,):
        super(unweightedVI, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

        self.enc = Encoder(s_dim, a_dim, z_dim, g_dim)
        self.dec = Decoder(s_dim, a_dim, z_dim)
        self.prior = torch.nn.Parameter(torch.zeros(2*z_dim))
        self.nu = 1e-2


    def loss(self, offline_data_m):
        z_mu_logvar = self.enc(offline_data_m)

        # reparametrization trick
        eps = torch.randn_like(z_mu_logvar[:self.z_dim])
        std = torch.exp(0.5 * z_mu_logvar[self.z_dim:])
        z = (eps*std+z_mu_logvar[:self.z_dim]) * torch.ones(offline_data_m.shape[0], self.z_dim)

        saz = torch.cat([offline_data_m[:,:(self.s_dim+self.a_dim)],z],dim=1)

        ds_mu_logvar = self.dec(saz)

        ret_loss = 0

        # y = offline_data[m,:,(s_dim+a_dim):-1]
        # mu = ds_mu_logvar[:,:s_dim]
        # logvar = ds_mu_logvar[:,s_dim:]
        ret_loss += gaussian_likelihood_loss(offline_data_m[:,(self.s_dim+self.a_dim):],
                                         ds_mu_logvar[:,:self.s_dim],
                                         ds_mu_logvar[:,self.s_dim:]) # approx of E_{z~q}[ - log p(y|x,z) ]

        ret_loss += self.nu * kld(z_mu_logvar[:self.z_dim],
                              z_mu_logvar[self.z_dim:],
                              self.prior[:self.z_dim],
                              self.prior[self.z_dim:]) # nu * E_{z~q}[ log q(z) - log p(z) ]
        return ret_loss
