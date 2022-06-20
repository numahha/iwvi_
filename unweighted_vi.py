import torch


def gaussian_likelihood_loss(y, mu, logvar):
    # If Var is diagonal matrix whose element isvar_i, then log(det|Var|) = log(prod_i var_i) = sum_i log(var_i)
    return 0.5 * torch.mean(((y-mu)**2) * torch.exp(-logvar) + logvar)


def kld(mu1, logvar1, mu2, logvar2):
    # kld(p1|p2) = E_{z~p1}[ log p1(z) - log p2(z) ]
    tmp1 = 0.5 * (logvar2 - logvar1) # log (sigma2/sigma1)
    tmp2 = 0.5*(torch.exp(logvar1)+(mu1-mu2)**2) / torch.exp(logvar2) # (sigma1^2+(mu1-mu2)^2)/(2*sigma2^2)
    return torch.mean(tmp1 + tmp2)


from enc_dec import Encoder, Decoder
class unweightedVI(torch.nn.Module):
    def __init__(self, s_dim, a_dim, z_dim, g_dim,):
        super(unweightedVI, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

        self.enc = Encoder(s_dim, a_dim, z_dim, g_dim)
        self.dec = Decoder(s_dim, a_dim, z_dim)        # approximate of posterior, p(z|X)
        self.prior = torch.nn.Parameter(torch.zeros(2*z_dim))
        self.nu = 1e0


    def loss(self, offline_data_m):
        z_mu_logvar = self.enc(offline_data_m[:, :(2*self.s_dim+self.a_dim)])

        # reparametrization trick
        eps = torch.randn_like(z_mu_logvar[:self.z_dim])
        std = torch.exp(0.5 * z_mu_logvar[self.z_dim:])
        z = (eps*std+z_mu_logvar[:self.z_dim]) * torch.ones(offline_data_m.shape[0], self.z_dim)

        saz = torch.cat([offline_data_m[:, :(self.s_dim+self.a_dim)],z],dim=1)

        ds_mu_logvar = self.dec(saz)

        ret_loss = 0

        ds = offline_data_m[:, (self.s_dim+self.a_dim):-1] - offline_data_m[:, :self.s_dim]
        ret_loss += gaussian_likelihood_loss(ds, # y
                                             ds_mu_logvar[:, :self.s_dim], # mu
                                             ds_mu_logvar[:, self.s_dim:]) # logvar # approx of E_{z~q}[ - log p(y|x,z) ]

        ret_loss += self.nu * kld(z_mu_logvar[:self.z_dim],
                                  z_mu_logvar[self.z_dim:],
                                  self.prior[:self.z_dim],
                                  self.prior[self.z_dim:]) # nu * E_{z~q}[ log q(z) - log p(z) ]
        return ret_loss


    def get_bamdpdata_from_mdpdata(self, offline_data_m):
        #s_data = offline_data_m[:, :self.s_dim]
        #a_data = offline_data_m[:, self.s_dim : (self.s_dim+self.a_dim)]
        #s_next_data = offline_data_m[:, (self.s_dim+self.a_dim) : (2*self.s_dim+self.a_dim)]

        b_next_data = []
        for n in range(len(offline_data_m)):
            b_next_data.append(self._get_belief(offline_data_m[:n+1, :(2*self.s_dim+self.a_dim)]))
        b_next_data = torch.vstack(b_next_data)
        b_data = torch.vstack([self._get_belief(), b_next_data[:-1]])
        # print(b_data.shape, b_next_data.shape)
        return torch.hstack([offline_data_m[:, :self.s_dim],
                             b_data,
                             offline_data_m[:, self.s_dim : (self.s_dim+self.a_dim)],
                             offline_data_m[:, (self.s_dim+self.a_dim) : (2*self.s_dim+self.a_dim)],
                             b_next_data,
                             offline_data_m[:, (2*self.s_dim+self.a_dim):]
                             ])


    def _get_belief(self, sads_array=None):
        if sads_array is None or len(sads_array)==0:
            return self.prior.detach()
        with torch.no_grad():
            return self.enc(sads_array)
