import torch
import numpy as np

class Bayes:

    def exact_posterior(data, l_cov, p_mean, p_cov, N):
        cov_inverse = (N) * torch.linalg.inv(l_cov) + torch.linalg.inv(p_cov)
        cov = torch.linalg.inv(cov_inverse)
        
        summation = torch.linalg.solve(p_cov, p_mean.unsqueeze(-1))
        for point in data:
            summation += torch.linalg.solve(l_cov, point.unsqueeze(-1))
            
        mean = torch.matmul(cov, summation)
        
        return mean.squeeze(1), cov

    def exact_posterior_1D(data, l_std, p_mean, p_std, N, scale=True):
        if scale:
            var_inv = (N) * (l_std ** -2) + (p_std ** -2)
            var = var_inv ** -1
            std = var ** 0.5

            num = p_mean / (p_std ** 2)
            for x in data:
                num += (x / (l_std ** 2))
            
            loc = num * var
        else:
            var_inv = (N) * (l_std ** -1) + (p_std ** -1)
            var = var_inv ** -1
            std = var 

            num = p_mean / (p_std)
            for x in data:
                num += (x / (l_std))
            
            loc = num * var
            
        return loc, std

    def maximum_likelihood(data):
        if len(data.shape) > 1:  
            N, D = data.shape[0], data.shape[1]
        else: 
            N, D = data.shape[0], 1
        loc = torch.zeros(D)
        for x in data:
            loc += x
        loc /= N

        if D > 1:
            cov = torch.zeros(D,D)
            for x in data:
                dif = x - loc
                cov += torch.matmul(dif.unsqueeze(1), dif.unsqueeze(0))
        else:
            cov = 0.0
            for x in data:
                cov += (x - loc) ** 2
        cov /= N
        
        return loc, cov