import torch, torchvision, torchaudio
import numpy as np

class Divergences:   
    
    def kl_gaussians(self, loc_1, loc_2, cov_1, cov_2):
        # Compute KL(p||q)= E_p(x)[log p(x) - log q(x)]
        assert loc_1.shape == loc_2.shape, "Not same dimension of means"
        assert cov_1.shape == cov_2.shape, "Not same Cov Matrix dimension"
        assert cov_1.shape[0] == cov_1.shape[1], "Not square matrix"

        d = cov_1.shape[0]
        diff = loc_1 - loc_2
        part_1 = torch.matmul(diff.unsqueeze(0), torch.linalg.solve(cov_2, diff.unsqueeze(-1))).squeeze()
        part_2 = torch.trace(torch.linalg.solve(cov_2, cov_1)).squeeze()
        part_3 = (torch.log(torch.linalg.det(cov_2)) - torch.log(torch.linalg.det(cov_1))).squeeze()

        #print(part_1, part_2, part_3, d)

        return (part_1 + part_2 + part_3 - d) / 2
        
    def reverse_kl(self, loc_1, loc_2, cov_1, cov_2):
        return self.kl_gaussians(loc_2, loc_1, cov_2, cov_1)
        
    def kl_spherical_gaussian(self, loc_1, loc_2, cov_1, cov_2, config):    
        
        diff = loc_2 - loc_1
        
        part_1 = torch.matmul(diff.unsqueeze(0), diff.unsqueeze(-1)).squeeze() / cov_2
        
        part_2 = config["D"] * cov_1 / cov_2
        
        part_3 = config["D"] * torch.log(cov_1 / cov_2)
        
        return (part_1 + part_2 - part_3 - config["D"]) / 2

    def reverse_kl_spherical(self, loc_1, loc_2, cov_1, cov_2, config):
        return self.kl_spherical_gaussian(loc_2, loc_1, cov_2, cov_1, config)
        
    def alpha_renyi(self, loc_1, loc_2, cov_1, cov_2, div_param):
        
        alpha = div_param
        
        diff = loc_1 - loc_2

        lin = (alpha * cov_2) + ((1 - alpha) * cov_1)
        
        part_1 = torch.matmul(diff.unsqueeze(0), torch.linalg.solve(lin, diff.unsqueeze(-1))).squeeze()
        
        part_2 = torch.log(torch.linalg.det(lin)).squeeze() / (alpha * (alpha - 1))
        
        part_3 = torch.log(torch.linalg.det(cov_1)).squeeze() / alpha
        
        part_4 = torch.log(torch.linalg.det(cov_2)).squeeze() / (1 - alpha)
        
        return (part_1 - part_2 - part_3 - part_4) / 2
        
    def alpha_renyi_spherical(self, loc_1, loc_2, var_1, var_2, config, div_param):
        
        diff = loc_1 - loc_2
        
        part_1_ = torch.matmul(diff.unsqueeze(0), diff.unsqueeze(-1)).squeeze()
        
        lin = (div_param * var_2 + (1 - div_param) * var_1)
        
        part_1 = part_1_ / (2 * lin)
        
        coef_1 = config["D"] / (2 * div_param)
        
        coef_2 = config["D"] / (2 * (div_param - 1))
        
        coef_3 = config["D"] / (2 * div_param * (div_param - 1))
        
        part_2 = coef_1 * torch.log(var_1)
        
        part_3 = coef_2 * torch.log(var_2)
        
        part_4 = coef_3 * torch.log(lin)
        
        return part_1 - part_2 + part_3 - part_4

    def gamma_spherical(self, loc_1, loc_2, var_1, var_2, config, div_param):
        
        var_1 = var_1
        var_2 = var_2
        
        D = config["D"]
        gamma = div_param
        
        diff = loc_1 - loc_2
        part_1_ = torch.matmul(diff.unsqueeze(0), diff.unsqueeze(-1)).squeeze()
        lin = (var_2 + (gamma - 1) * var_1)
        part_1 = (part_1_ / (2 * lin)).squeeze()
        
        part_2 = (D / (2*gamma - 2)) * torch.log(lin).squeeze()
        
        part_3 = (D / (gamma -1)) * torch.log(var_1).squeeze()
        
        #Only parts 1-3 are relevant for optimisation but taking out the constants might make convergence metrics
        #have different numbers that the ELBO converges to
        
        part_4 = (gamma/(gamma-1))*torch.log(torch.tensor(2*np.pi))
        
        part_5 = ((D*gamma)/(2*gamma - 2))*torch.log(var_2).squeeze()
        
        part_6 = (D/2)*torch.log(torch.tensor(2*np.pi)).squeeze()
        
        part_7 = (D/(2*gamma - 2))*torch.log(torch.tensor(gamma)).squeeze()
        
        const = part_6 - part_7 - part_5 - part_4
        #const = 0.
        
        return part_1 + part_2 + part_3 + const
        
    def FisherRao_normals(self, mu_1, mu_2, sigma_1_sq, sigma_2_sq):
        sigma_1 = sigma_1_sq ** 0.5
        sigma_2 = sigma_2_sq ** 0.5
        
        diff_mu = (mu_2-mu_1) ** 2
        neg_sigma = (sigma_2-sigma_1) ** 2
        pos_sigma = (sigma_2+sigma_1) ** 2
        numerator = diff_mu + 2 * neg_sigma
        denominator = diff_mu + 2 * pos_sigma
        Delta = (numerator / denominator) ** 0.5
        
        frac = (1 + Delta) / (1-Delta)
        
        FR = (2 ** 0.5) * (frac.log())
        
        return FR