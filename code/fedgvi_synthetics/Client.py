from __future__ import division
import copy


import torch
import torch.utils.data
import torch.nn as nn
from torch import distributions, optim
import numpy as np

from fedgvi_synthetics.Divergences import Divergences
from fedgvi_synthetics.Bayes import Bayes
from fedgvi_synthetics.helper_functions import helper_functions

JITTER = 1e-8

device = 'cpu'
torch.set_default_dtype(torch.float64)
#torch.set_default_device(device = device)

torch.manual_seed(86_960_947)


class Client:    

    def __init__(self, client=None, config=None):
        if config is None:
            self.config = self.get_default_config()
        else:
            self.config = config
        
        default_client = self.get_default_client()
        if client is None:
            self.client = default_client
        else:
            keys = client.keys()
            default_keys = default_client.keys()
            for k in default_keys:
                if k in keys:
                    default_client.update({k : client[k]})
            
            self.client = default_client

    def help(self):
        return {
            "client_idx": "Client Index, integer",
            "x_n": "Data set, torch tensor",
            "mean": "torch tensor, initial mean of approximate loss, default is torch.tensor([0.], device = device)",
            "variance": "torch tensor, initial variance of approximate loss default is torch.tensor([np.inf], device=device)",
            "variance_inverse": "Inverse of 'variance', default is torch.tensor([0.], device=device)",
            "normaliser": "Normalising constant of client, is not required for model",
            "iteration": "Current iteration of client, default is 0",
            "true_v": "Variance of assumed DGP, assumed to be Gaussian",
            "spherical": "Is this a spherical/univariate distribution",
            "require_s_n": "Do we require a normalising constant for the approximate loss, default is False",
            "Divergence": "Client divergence, options are: 'KLD', 'wKL', 'RKL', 'AR', 'G', 'FR'",
            "div_param": "Divergence hyperparameter, real number",
            "loss": "Loss type used, options are: 'nll', 'beta', 'gamma', 'score_matching'",
            "score_function": "weighting function for score_matching loss, options are 'IMQ', 'SE'",
            "loss_param": "Float with loss hyper parameter",
        }
    
    def get_default_config(self):
        return {
            "epochs": 20,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 0.05},
            "model_optimiser_params": {},
            "sigma_optimiser_params": {},
            "lr_scheduler": "MultiplicativeLR",
            "lr_scheduler_params": {"lr_lambda": lambda epoch: 1.0},
        }
    
    def get_default_client(self):
        return {
            "client_idx": 0,
            "x_n": [],
            "mean": torch.tensor([0.], device = device),
            "variance": torch.tensor([np.inf], device=device),
            "variance_inverse": torch.tensor([0.], device=device),
            "normaliser": torch.tensor([1.], device = device),
            "iteration": 0,
            "true_v": torch.tensor([1.], device=device),
            "spherical": True,
            "require_s_n": False,
            "Divergence": "KLD",
            "div_param": None,
            "loss": "nll",
            "score_function": None,
            "loss_param": None,
        }
    
    def update_q(self, q, parameters):

        if self.config["conjugate"]:
            q_new, t_new = self.conjugate_update(q)
        else:
            q_new, t_new = self.gradient_based_update(q, parameters)
            
        return q_new, t_new

    def gradient_based_update(self, q_s, parameters):
        q_old = copy.deepcopy(q_s)
        q_cav = self.cavity(q_s)
        
        q = {
            "loc": torch.nn.Parameter(copy.deepcopy(q_s["loc"].detach())),
            "var": torch.nn.Parameter(torch.log(copy.deepcopy(q_s["var"].detach()))),
        }

        q_params = list(q.items())

        q_parameters = [ 
                        {"params": q["loc"]},
                        {"params": q["var"]}
                    ]    
        
        #print(q_parameters)
        optimiser = optim.Adam(q_parameters, lr=parameters["lr"])
        #print(optimiser)
        
        lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            optimiser, **self.config["lr_scheduler_params"])
        
        #epoch_iter = tqdm(
        #    range(config["epochs"]),
        #    desc="Epoch",
        #    leave=True,
        #)
        for i in range(self.config["optim_epochs"]):
            
            # Calculate the minimization step and iteratively refine the mean and variance parameters
            # Compute argmin(-F(q)) = argmin {KL(q||q_cav) - Sum(E_q[log p(y_k_i | theta)])}
            
            #epoch = defaultdict(lambda: 0.0)
            """if client["iteration"] > 0:
                print(i)
                print(f"q_cav: loc: ",q_cav["loc"], " and var: ",q_cav["var"])
                print(f"Client: loc: ", client["mean"]," var: ", client["variance"], " var_inverse: ", client["variance_inverse"])
                print(f"q current: loc: ", q["loc"].detach(), " and var: ", q["var"].detach())"""
            optimiser.zero_grad()
            
            coef = 1.
            batch = self.client["x_n"]
            if self.config["minibatch"]:
                coef = len(self.client["x_n"]) / self.config["batchsize"]
                batch = helper_functions.get_batch(self.client["x_n"], self.config["batchsize"])
            
            if self.client["spherical"]:
                if self.client["Divergence"] == "KLD":
                    div = Divergences().kl_spherical_gaussian(q["loc"], q_cav["loc"], torch.exp(q["var"]), q_cav["var"], self.config)
                elif self.client["Divergence"] == "RKL":
                    div =  Divergences().reverse_kl_spherical(q["loc"], q_cav["loc"], torch.exp(q["var"]), q_cav["var"], self.config)
                elif self.client["Divergence"] == "wKL":
                    div =  Divergences().kl_spherical_gaussian(q["loc"], q_cav["loc"], torch.exp(q["var"]), q_cav["var"], self.config) / self.client["div_param"]
                elif self.client["Divergence"] == "AR":
                    div =  Divergences().alpha_renyi_spherical(q["loc"], q_cav["loc"], torch.exp(q["var"]), q_cav["var"], self.config, self.client["div_param"])
                elif self.client["Divergence"] == "G":
                    div =  Divergences().gamma_spherical(q["loc"], q_cav["loc"], torch.exp(q["var"]), q_cav["var"], self.config, self.client["div_param"])
                else:
                    if self.config["is_1d"] and self.client["Divergence"] == "FR":
                        div =  Divergences().FisherRao_normals(q["loc"], q_cav["loc"], torch.exp(q["var"]), q_cav["var"])
                    else:
                        print("!!! No Divergence specified !!!")
                        div = 0
            else:
                if self.client["Divergence"] == "KLD":
                    div =  Divergences().kl_gaussians(q["loc"], q_cav["loc"], torch.exp(q["var"]), q_cav["var"])
                elif self.client["Divergence"] == "RKL":
                    div =  Divergences().reverse_kl(q["loc"], q_cav["loc"], torch.exp(q["var"]), q_cav["var"])
                elif self.client["Divergence"] == "wKL":
                    div =  Divergences().kl_gaussians(q["loc"], q_cav["loc"], torch.exp(q["var"]), q_cav["var"]) / self.config["local_div_param"]
                elif self.client["Divergence"] == "AR":
                    div =  Divergences().alpha_renyi(q["loc"], q_cav["loc"], torch.exp(q["var"]), q_cav["var"], self.client["div_param"])
                else:
                    print("!!! No Divergence specified !!!")
                    div = 0
            
            ll = 0.0

            if self.client["loss"] == "beta" or self.client["loss"] == "gamma":
                loss_param = self.client["loss_param"]
                integral_term = divergence_losses.integral_div_loss(self.client["true_v"], loss_param, self.config["D"])
                for y in batch:
                    expectation_term = divergence_losses.expectation_div_loss(q, y, self.client["true_v"], loss_param - 1, self.config["D"])
                    #print("Expectaton Term: ", expectation_term, " and integral term: ", integral_term)
                    if self.client["loss"] == "beta":
                        ll += (expectation_term + (integral_term / loss_param ))
                    else:
                        ll += (expectation_term * loss_param / (integral_term ** ((loss_param - 1)/ loss_param)))
                        
            elif self.client["loss"] == "nll": # Negative log likelihood
                for x in batch:
                    dif = q["loc"] - x
                    if self.client["spherical"]:
                        if self.config["is_1d"]:
                            ll += (dif ** 2) / (self.client["true_v"] * 2)
                            ll +=  torch.exp(q["var"]) / (2 * self.client["true_v"])
                        else:
                            ll += torch.matmul(dif.unsqueeze(0), torch.linalg.solve(self.client["true_v"], dif.unsqueeze(-1))).squeeze() / 2
                            ll +=  torch.exp(q["var"]) / (2 * self.client["true_v"])
                    else:
                        ll += torch.matmul(dif.unsqueeze(0), torch.linalg.solve(self.client["true_v"], dif.unsqueeze(-1))).squeeze() / 2
                        ll += torch.trace(torch.linalg.solve(self.client["true_v"], torch.exp(q["var"]))) / 2
                    
            else: # Default: Euclidean distance  
                for x in batch:
                    dif = q["loc"] - x
                    ll += torch.linalg.vector_norm(dif)
            
            if coef < 1: 
                ll = ll * coef

            #ll /= len(samples)
            
            loss = ll + div
            #print(ll, div, loss)
            #break
            loss.backward()
            optimiser.step()
            
            """if i == (config["optim_epochs"]-1):
                q["metrics"]["elbos"].append(-loss.item())
                q["metrics"]["kls"].append(div.item())
                #q["metrics"]["lls"].append(ll.item())
                q["metrics"]["lls"].append(0.)"""

        v_temp = torch.exp(copy.deepcopy(q["var"].detach()))
        q.update({"var": nn.Parameter(v_temp)})

        t_new = self.update_client_t(q, q_old)
        lr_scheduler.step()
        print("Client ", self.client["client_idx"], " updates as: ", t_new, "\n With cavity distribution: ", q_cav, " and current approx: ", q["loc"], q["var"])
        return q, t_new
    
    #If we use the KL divergence and we are conjugate, we can find explicit updates
    # This takes the form: q_new = argmin_q {E_q[-log p(x_m|θ)]+KL(q:q_cav)}
    def conjugate_update(self, q):
        print("Conjugate Update:")
        q_old = copy.deepcopy(q)
        q_cav = self.cavity(q)
        print("Cavity distribution: ", q_cav)
        N = len(self.client["x_n"])
        if self.client["spherical"] or self.config["is_1d"]:
            if self.client["loss"] == "score_matching":
                # weighted Score matching as in Altamirano et al. (2024)
                assert self.config["is_1d"], "Trying to use conjugate score matching loss with multivariate distribution!\n Not yet implemented."

                beta = ((self.client["true_v"] / 2) ** 0.5)
                beta.to(device)
                c = self.client["loss_param"]
                
                w_scores_squared = 0.0 

                q_loc = q_cav["loc"] / q_cav["var"]
                
                for data_point in self.client["x_n"]:
                    if self.client["score_function"] == "IMQ":
                        w_score, w_score_dy = weighting_functions.IMQ(data_point, q_cav["loc"], beta, c)
                    elif self.client["score_function"] == "SE":
                        w_score, w_score_dy = weighting_functions.SE(data_point, q_cav["loc"], beta, c)
                    else:
                        print("No weighting function specified")
                        w_score, w_score_dy = 1 , 0
                    
                    wscore_squared = w_score ** 2
                    w_scores_squared += wscore_squared
                    
                    log_wscore_dy = 2 * w_score_dy / w_score
                    
                    part = data_point - self.client["true_v"] * log_wscore_dy

                    q_loc += ((2 * wscore_squared * part) / (self.client["true_v"] ** 2))
                
                q_var = ((q_cav["var"] ** -1) + ((2 * w_scores_squared) / (self.client["true_v"] ** 2))) ** -1

                q_loc *= q_var

                q_new = {
                        "loc": q_loc,
                        "var": q_var
                    }
            else: #negative log likelihood
                if self.config["is_1d"]:
                    # The below assumes the standard deviation instead of the variance so we take the square root
                    q_loc, q_var = Bayes.exact_posterior_1D(self.client["x_n"], self.client["true_v"], q_cav["loc"], q_cav["var"], N, scale=False)
                    q_new = {
                        "loc": q_loc,
                        "var": q_var
                    }
                else:
                    q_loc, q_var = Bayes.exact_posterior(self.client["x_n"], self.client["true_v"]*torch.eye(self.config["D"]), q_cav["loc"], q_cav["var"]*torch.eye(self.config["D"]), N)
                    q_new = {
                        "loc": q_loc,
                        "var": q_var[0][0] ** 2
                    }
            print("New approximation: loc: ", q_new["loc"]," var: ", q_new["var"])
        else:
            q_loc, q_var = Bayes.exact_posterior(self.client["x_n"], self.client["true_v"], q_cav["loc"], q_cav["var"], N)

            q_new = {
                "loc": q_loc,
                "var": q_var
            }
        q.update({"loc": q_new["loc"], "var": q_new["var"]})
        print("This should match the previous: \n", "Returned client approximation: loc: ", q_new["loc"]," var: ", q_new["var"])
        t_new = self.update_client_t(q, q_old)
        print("New client likelihood: ", t_new)
        return q, t_new

    def cavity(self, q):
        if self.client["spherical"]:
            v_bar_n_inverse = q["var"].detach() ** (-1) - self.client["variance_inverse"]
            v_bar_n = v_bar_n_inverse ** (-1)

            part = (q["var"].detach() ** (-1)) * q["loc"].detach() - self.client["variance_inverse"] * self.client["mean"]

            m_bar_n = v_bar_n * part
        else:
            v_bar_n_inverse = torch.linalg.inv(q["var"].detach()) - self.client["variance_inverse"]
            v_bar_n = torch.linalg.inv(v_bar_n_inverse)
            
            part = torch.linalg.solve(q["var"].detach(), q["loc"].detach().unsqueeze(-1)) - torch.matmul(self.client["variance_inverse"], self.client["mean"].unsqueeze(-1))
            m_bar_n = torch.linalg.solve(v_bar_n_inverse, part).squeeze(-1)
        print("Cavity: loc: ", m_bar_n, " var: ", v_bar_n)
        return {
            "loc": m_bar_n,
            "var": v_bar_n
        }
        
    #We udate t according to the product of experts formula for gaussians
    def update_client_t(self, q, q_old):

        old_var = self.client["variance"]
        old_var_inv = self.client["variance_inverse"]
        old_mean = self.client["mean"]
        
        if self.client["iteration"] == 0:
            if self.client["spherical"] != True:
                Sigma = q["var"].detach()
                Lambda = q_old["var"].detach()
                
                mu = q["loc"].detach().unsqueeze(-1)
                eta = q_old["loc"].detach().unsqueeze(-1)
            
                Sigma_inv = torch.linalg.inv(Sigma)
                Lambda_inv = torch.linalg.inv(Lambda)
                
                variance_inverse = Sigma_inv - Lambda_inv
                
                variance = torch.linalg.inv(variance_inverse)
                
                mean_ = (torch.linalg.solve(Sigma, mu) - torch.linalg.solve(Lambda, eta))
                
                mean = torch.matmul(variance, mean_).squeeze(-1)
                s_n = None
            else:
                variance_inverse = (q["var"].detach() ** -1) - (q_old["var"].detach() ** -1)
                variance = variance_inverse ** -1
                
                mean = variance * ((q["var"].detach() ** -1) * q["loc"].detach() - (q_old["var"].detach() ** -1) * q_old["loc"].detach())
                s_n = None
                if variance <= 0:
                    print("!!! Variance negative!!!")
        else:
            if self.client["spherical"] != True:
                Sigma = q["var"].detach()
                Lambda = q_old["var"].detach()
                
                mu = q["loc"].detach().unsqueeze(-1)
                eta = q_old["loc"].detach().unsqueeze(-1)
            
                Sigma_inv = torch.linalg.inv(Sigma)
                Lambda_inv = torch.linalg.inv(Lambda)
                
                variance_inverse = Sigma_inv - Lambda_inv + old_var_inv
                
                variance = torch.linalg.inv(variance_inverse)
                
                mean_ = (torch.linalg.solve(Sigma, mu) - torch.linalg.solve(Lambda, eta) +
                       torch.matmul(old_var_inv, old_mean.unsqueeze(-1)))
                
                mean = torch.matmul(variance, mean_).squeeze(-1)
                
                #print(f"New q: Sigma: {Sigma}, and Sigma_inv: {Sigma_inv}, and loc: {mu}")
                #print(f"Old q: Lambda: {Lambda}, and Lambda_inv: {Lambda_inv}, and loc: {eta}")
                
                s_n = None
            else:
                variance_inverse = ((q["var"].detach() ** -1) - 
                                    (q_old["var"].detach() ** -1) + self.client["variance_inverse"])
                variance = variance_inverse ** -1
                
                mean = variance * ((q["var"].detach() ** -1) * q["loc"].detach() -
                       (q_old["var"].detach() ** -1) * q_old["loc"].detach() +
                       self.client["variance_inverse"] * self.client["mean"])
                
                s_n = None
                if variance <= 0:
                    print("!!! Variance negative!!!")
                
            #print("variance_inverse: ", variance_inverse)
            #print("variance: ", variance)
            #print("mean: ", mean)
            
        return {
            "mean": mean,
            "variance": variance,
            "variance_inverse": variance_inverse,
            "normaliser": s_n
        }

# Weighted Score matching as in Altamirano et al. (2024)
class weighting_functions:
    def IMQ(y, x, beta, c, alpha=0.5):
        dif = y - x
        return weighting_functions.IMQ_wscore(dif, beta, c, alpha), weighting_functions.IMQ_wscore_dy(dif, beta, c, alpha)
        
    def SE(y, x, beta, c):
        dif = y - x
        w_score_temp = weighting_functions.SE_wscore(dif, beta, c)
        return w_score_temp, weighting_functions.SE_wscore_dy(dif, c, w_score_temp)
    
    def IMQ_wscore(y, beta, c, alpha):
        return beta * ((1 + ((y ** 2) / (2 * alpha * (c ** 2)))) ** (-alpha))

    def IMQ_wscore_dy(y, beta, c, alpha):
        return - beta * ((1 + ((y ** 2) / (2 * alpha * (c ** 2)))) ** (-alpha-1)) * y / (c ** 2)

    def SE_wscore(y, beta, c):
        return beta * torch.exp(- 0.5 * ((y/c) ** 2))
    
    def SE_wscore_dy(y, c, w_score):
        return - w_score * y / (c ** 2)

class divergence_losses:
    #Beta/ Gamma loss terms for assumed Gaussian likelihoods as in Knoblauch et al. (2022)
    def integral_div_loss(var, c, d):
        a = torch.linalg.det(2 * np.pi * var) if len(var.shape) > 1 else torch.linalg.det(2 * np.pi * var * torch.eye(d, device=device))
        return ((a ** c) * (c ** d)) ** (-0.5)

    def expectation_div_loss(q, y, var, c, d):
        if d == 1:
            return divergence_losses.expectation_div_loss_1d(q, y, var, c)
        elif d > 1:
            if var.shape[0] == d:
                return divergence_losses.expectation_div_loss_matrix(q, y, var, c)
            else:
                return divergence_losses.expectation_div_loss_matrix(q, y, var * torch.eye(d, device=device), c)
        else:
            print("Covariance does not have positive dimension")
            return None
    
    def expectation_div_loss_1d(q, y, var, c):
        lambda_tilde = (c/var + 1/torch.exp(q["var"])) ** -1
        mu_tilde = (c*y/var + q["loc"]/torch.exp(q["var"]))

        part1 = (2 * np.pi * var) ** (-c/2)
        part2 = torch.sqrt(lambda_tilde/torch.exp(q["var"]))
        part3 = - ((c * (y ** 2) / var) + ((q["loc"] ** 2)/ torch.exp(q["var"])) - ((mu_tilde ** 2) * lambda_tilde)) / 2
        
        return - part1 * part2 * torch.exp(part3) / c
    
    def expectation_div_loss_matrix(q, y, var, c):
        lambda_tilde = torch.linalg.inv(c * torch.linalg.inv(var) + torch.linalg.inv(torch.exp(q["var"])))
        mu_tilde = c * torch.linalg.solve(var, y.unsqueeze(-1)) + torch.linalg.solve(torch.exp(q["var"]), q["loc"].unsqueeze(-1))
        
        part1 = torch.linalg.det(2 * np.pi * var) ** (-c/2)
        part2 = torch.linalg.det(lambda_tilde) / torch.linalg.det(torch.exp(q["var"]))
        part3 = - ((c * torch.matmul(y.unsqueeze(0), torch.linalg.solve(var, y.unsqueeze(-1))).squeeze()) + 
                   (torch.matmul(q["loc"].unsqueeze(0), torch.linalg.solve(torch.exp(q["var"]), q["loc"].unsqueeze(-1)))) - 
                   (torch.matmul(mu_tilde.T, torch.linalg.inv(lambda_tilde, mu_tilde)))) / 2
        
        return - part1 * torch.sqrt(part2) * torch.exp(part3) / c