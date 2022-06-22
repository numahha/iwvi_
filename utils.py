import math
import torch


def gaussian_likelihood_loss(y, mu, logvar):
    # If Var is diagonal matrix whose element isvar_i, then log(det|Var|) = log(prod_i var_i) = sum_i log(var_i)
    return 0.5 * torch.mean(((y-mu)**2) * torch.exp(-logvar) + logvar)

def kld(mu1, logvar1, mu2, logvar2):
    # kld(p1|p2) = E_{z~p1}[ log p1(z) - log p2(z) ]
    tmp1 = 0.5 * (logvar2 - logvar1) # log (sigma2/sigma1)
    tmp2 = 0.5*(torch.exp(logvar1)+(mu1-mu2)**2) / torch.exp(logvar2) # (sigma1^2+(mu1-mu2)^2)/(2*sigma2^2)
    return torch.mean(tmp1 + tmp2)

def kdl_var_approx(mu1, logvar1, mu2_list, logvar2_list):
    # Eq (18) in Lower and upper bounds for approximation of the Kullback-Leibler divergence between Gaussian mixture models (2012)
    # Assume f is single gaussian, while g is a mixture of gaussian with uniform weights
    # f = N(mu1,logvar1), g = (1/M) * sum_m N(mu2_list[m], logvar2_list[m])
    # Under this assumption, kld_fa_falph=1, leading to numerator=1 in Eq (18).

    M = len(mu2_list)
    denominator = 0
    for m in range(M):
        kld_fa_gb = kld(mu1, logvar1, mu2_list[m], logvar2_list[m])
        #print("kld_fa_gb",kld_fa_gb)
        denominator += torch.exp(-kld_fa_gb)
    #print("de",denominator)
    denominator /= M
    log_numerator_denominator = - torch.log(denominator + 1e-30)
    return log_numerator_denominator


# for sac
def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
