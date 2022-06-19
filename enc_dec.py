import torch


class Encoder(torch.nn.Module):
    def __init__(self, s_dim, a_dim, z_dim, g_dim):
        super(Encoder, self).__init__()
        self.net_f = torch.nn.Sequential(
                            torch.nn.Linear(2*s_dim+a_dim, 64),
                            torch.nn.ReLU(),
                            torch.nn.Linear(64, g_dim),
                            )
        self.net_q = torch.nn.Sequential(
                            torch.nn.Linear(g_dim, 64),
                            torch.nn.ReLU(),
                            torch.nn.Linear(64, 2*z_dim),
                            )

    def forward(self, data_m):
        g_m = self.net_f(data_m).mean().reshape(1,-1)
        return self.net_q(g_m).flatten()


class Decoder(torch.nn.Module):
    def __init__(self, s_dim, a_dim, z_dim):
        super(Decoder, self).__init__()
        self.net_phat = torch.nn.Sequential(
                            torch.nn.Linear(s_dim+a_dim+z_dim, 64),
                            torch.nn.ReLU(),
                            torch.nn.Linear(64, 2*s_dim),
                            )

    def forward(self, saz):
        return self.net_phat(saz)
