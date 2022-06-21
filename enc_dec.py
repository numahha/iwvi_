import torch


class Encoder(torch.nn.Module):
    def __init__(self, s_dim, a_dim, z_dim, g_dim):
        super(Encoder, self).__init__()
        h_dim=64
        activate_fn=torch.nn.Tanh
        self.net_f = torch.nn.Sequential(
                            torch.nn.Linear(2*s_dim+a_dim, h_dim),
                            activate_fn(),
                            torch.nn.Linear(h_dim, h_dim),
                            activate_fn(),
                            torch.nn.Linear(h_dim, g_dim),
                            )
        self.net_q = torch.nn.Sequential(
                            torch.nn.Linear(g_dim, h_dim),
                            activate_fn(),
                            torch.nn.Linear(h_dim, h_dim),
                            activate_fn(),
                            torch.nn.Linear(h_dim, 2*z_dim),
                            )

    def forward(self, data_m):
        #g_m = self.net_f(data_m).mean(0).reshape(1,-1)
        g_m = self.net_f(data_m).sum(0).reshape(1,-1)
        return self.net_q(g_m).flatten()


class Decoder(torch.nn.Module):
    def __init__(self, s_dim, a_dim, z_dim):
        super(Decoder, self).__init__()
        h_dim=64
        activate_fn=torch.nn.Tanh
        self.net_phat = torch.nn.Sequential(
                            torch.nn.Linear(s_dim+a_dim+z_dim, h_dim),
                            activate_fn(),
                            torch.nn.Linear(h_dim, h_dim),
                            activate_fn(),
                            torch.nn.Linear(h_dim, 2*s_dim),
                            )

    def forward(self, saz):
        return self.net_phat(saz)
