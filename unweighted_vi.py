import numpy as np
import torch

from utils import gaussian_likelihood_loss, kld, kdl_var_approx
from enc_dec import Encoder, Decoder


class unweightedVI(torch.nn.Module):
    def __init__(self, offline_data, s_dim, a_dim, z_dim, g_dim,):
        super(unweightedVI, self).__init__()

        self.offline_data = offline_data # [M, N , |SAS'R|] : M ... num of MDPs, N ... trajectory length, |SAS'R| ... dim of (s,a,s',r)
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

        self.enc = Encoder(s_dim, a_dim, z_dim, g_dim)         # q(z|D^train_m)
        self.enc_belief = Encoder(s_dim, a_dim, z_dim, g_dim)  # beta^t(z)
        self.dec = Decoder(s_dim, a_dim, z_dim)                # p(s'-s|s,a,z)
        self.prior = torch.nn.Parameter(torch.zeros(2*z_dim))  # [mean, logvar]
        self.initial_belief = torch.nn.Parameter(torch.zeros(2*z_dim))  # [mean, logvar]
        self.nu = 1e0



    def metatrain_model(self,num_iter=1000, lr=1e-4):

        for p in self.enc.parameters():
            p.requires_grad = True
        for p in self.dec.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.parameters(),lr=lr)


        for i in range(num_iter):
            m = np.random.randint(self.offline_data.shape[0])
            optimizer.zero_grad()
            loss = self._loss_metatrain_model(self.offline_data[m,:,:])
            loss.backward()
            optimizer.step()
            if 0==i%10:
                print("metatrain_model: iter",i," loss",loss.item())


        self.belief_mu_list = []
        self.belief_logvar_list = []
        with torch.no_grad():
            for m in range(len(self.offline_data)):
                mu_logvar = self.enc(self.offline_data[m, :, :(2*self.s_dim+self.a_dim)])
                self.belief_mu_list.append(mu_logvar[:self.z_dim])
                self.belief_logvar_list.append(mu_logvar[self.z_dim:])



    def _loss_metatrain_model(self, offline_data_m):
        z_mu_logvar = self.enc(offline_data_m[:, :(2*self.s_dim+self.a_dim)])

        # reparametrization trick
        eps = torch.randn_like(z_mu_logvar[:self.z_dim])
        std = torch.exp(0.5 * z_mu_logvar[self.z_dim:])
        z = (eps*std+z_mu_logvar[:self.z_dim]) * torch.ones(offline_data_m.shape[0], self.z_dim)

        saz = torch.cat([offline_data_m[:, :(self.s_dim+self.a_dim)], z], dim=1)

        ds_mu_logvar = self.dec(saz)
        ds = offline_data_m[:, (self.s_dim+self.a_dim):-1] - offline_data_m[:, :self.s_dim]

        loss = 0
        # Approximate of E_{z~q}[ - log p(y|x,z) ]
        loss += gaussian_likelihood_loss(ds, # y
                                         ds_mu_logvar[:, :self.s_dim], # mu
                                         ds_mu_logvar[:, self.s_dim:]) # logvar
        # nu * E_{z~q}[ log q(z) - log p(z) ]
        loss += self.nu * kld(z_mu_logvar[:self.z_dim],
                              z_mu_logvar[self.z_dim:],
                              self.prior[:self.z_dim],
                              self.prior[self.z_dim:])
        return loss



    def train_enc_belief(self, num_iter=1000, lr=1e-4):

        for p in self.enc.parameters():
            p.requires_grad = False
        for p in self.dec.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam(self.parameters(),lr=lr)

        for i in range(num_iter):
            m = np.random.randint(self.offline_data.shape[0])
            optimizer.zero_grad()
            loss = self._loss_train_enc_belief(self.offline_data[m,:,:])
            loss.backward()
            optimizer.step()
            if 0==i%10:
                print("train_enc_belief: iter",i," loss",loss.item())



    def _loss_train_enc_belief(self, offline_data_m):
        z_mu_logvar = self.enc_belief(offline_data_m[:, :(2*self.s_dim+self.a_dim)])

        # reparametrization trick
        eps = torch.randn_like(z_mu_logvar[:self.z_dim])
        std = torch.exp(0.5 * z_mu_logvar[self.z_dim:])
        z = (eps*std+z_mu_logvar[:self.z_dim]) * torch.ones(offline_data_m.shape[0], self.z_dim)

        saz = torch.cat([offline_data_m[:, :(self.s_dim+self.a_dim)], z], dim=1)
        ds_mu_logvar = self.dec(saz)
        ds = offline_data_m[:, (self.s_dim+self.a_dim):-1] - offline_data_m[:, :self.s_dim]

        loss = 0
        # Approximate of E_{z~q}[ - log p(y|x,z) ]
        loss += gaussian_likelihood_loss(ds, # y
                                         ds_mu_logvar[:, :self.s_dim], # mu
                                         ds_mu_logvar[:, self.s_dim:]) # logvar
        #print("loss",loss)
        # nu * E_{z~q}[ log q(z) - log p(z) ]
        loss += self.nu * kdl_var_approx(z_mu_logvar[:self.z_dim],
                              z_mu_logvar[self.z_dim:],
                              self.belief_mu_list,
                              self.belief_logvar_list)
        #print("loss",loss)
        return loss


    def train_initial_belief(self, num_iter=1000, lr=1e-4):
        optimizer = torch.optim.Adam(self.parameters(),lr=lr)

        for i in range(num_iter):
            m = np.random.randint(self.offline_data.shape[0])
            optimizer.zero_grad()
            loss = kdl_var_approx(self.initial_belief[:self.z_dim],
                                  self.initial_belief[self.z_dim:],
                                  self.belief_mu_list,
                                  self.belief_logvar_list)
            loss.backward()
            optimizer.step()
            if 0==i%10:
                print("train_initial_belief: iter",i," loss",loss.item())



    def get_belief(self, sads_array=None):
        with torch.no_grad():
            if sads_array is None or len(sads_array)==0:
                return self.initial_belief.detach()
            else:
                return self.enc(sads_array)


    def get_bamdpdata_from_mdpdata(self, offline_data_m):
        #s_data = offline_data_m[:, :self.s_dim]
        #a_data = offline_data_m[:, self.s_dim : (self.s_dim+self.a_dim)]
        #s_next_data = offline_data_m[:, (self.s_dim+self.a_dim) : (2*self.s_dim+self.a_dim)]

        b_next_data = []
        for n in range(len(offline_data_m)):
            b_next_data.append(self.get_belief(offline_data_m[:n+1, :(2*self.s_dim+self.a_dim)]))
        b_next_data = torch.vstack(b_next_data)
        b_data = torch.vstack([self.get_belief(), b_next_data[:-1]])
        # print(b_data.shape, b_next_data.shape)
        return torch.hstack([offline_data_m[:, :self.s_dim], # a
                             b_data,
                             offline_data_m[:, self.s_dim : (self.s_dim+self.a_dim)], # a
                             offline_data_m[:, (self.s_dim+self.a_dim) : (2*self.s_dim+self.a_dim)], # s_next
                             b_next_data,
                             offline_data_m[:, (2*self.s_dim+self.a_dim):] # r
                             ])
    #
    #
    #
    #
    # def step(self, a):
    #     return 0
    #
    # def reset(self):
    #     return 0
