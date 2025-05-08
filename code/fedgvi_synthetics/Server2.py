from __future__ import division
import copy


import torch
import torch.utils.data
import torch.nn as nn
from torch import distributions, optim
from torchvision import transforms, datasets
import scipy.stats as stats
import numpy as np


JITTER = 1e-8

device = 'cpu'
torch.set_default_dtype(torch.float64)
#torch.set_default_device(device = device)

torch.manual_seed(86_960_947)

from fedgvi_synthetics.Client import Client
from fedgvi_synthetics.helper_functions import helper_functions
from fedgvi_synthetics.Divergences import Divergences
from fedgvi_synthetics.Bayes import Bayes

class Server2:
    #This method has the server optimisation step and prior refinement integrated

    def help(self):
        return Client().help(), self.server_help()

    def server_help(self):
        return {
            "a": "True (co)variance of DGP",
            "D": "Dimension of data",
            "N": 'Number of data points in entire data set',
            "Epochs": "Number of client iterations",
            "global_div": "Server divergence, options are: 'KLD', 'wKL', 'RKL', 'AR', 'G', 'FR'",
            "global_div_param": "Server divergence hyper-parameter",
            "lr": "Torch optimiser learning rate, default '1e-3'",
            "samples": "If using Monte Carlo approximation, the number of samples drawn",
            "optim_epochs": "Number of optimisation Epochs",
            "batch_size": "If using minibatches, this is the size of them"   
        }

    def FedGVI(self, q_global, clients, parameters, is_1d=False, minibatch=False, conjugate=False):
        D = parameters["D"]
        
        global_div = parameters["global_div"]
        div_hyper_param = parameters["global_div_param"]
        
        if minibatch:
            batchsize = parameters["batch_size"]
        else:
            batchsize = np.inf
            
        prior = {
            "loc": q_global["loc"].detach(),
            "var": q_global["var"].detach(),
        }
            
        config = {
            "D": D,
            "epochs": parameters["Epochs"], 
            "samples": parameters["samples"],
            "lr_scheduler_params": {"lr_lambda": lambda epoch: 1.0},
            "optim_epochs": parameters["optim_epochs"],
            "lr": parameters["lr"],
            "is_1d": is_1d,
            "early_stopping": False,
            "minibatch": minibatch,
            "batchsize": batchsize,
            "global_div": global_div,
            "div_hyper_param": div_hyper_param,
            "conjugate": conjugate
        }
        
        print("Starting global q: ", q_global)
        
        prior_list = [copy.deepcopy(prior)]
        q_list = [copy.deepcopy(prior)]
        
        for i in range(config["epochs"]):
            print("===========================================")
            print(f"Iteration {i+1}:")
            print(q_global["loc"], q_global["var"])
            
            q_global_list = []
            for n in range(len(clients)):
                temp = copy.deepcopy(q_global)
                q_global_list.append(temp)
            
            for n in range(len(clients)):
                print(f"Client {n}")
                client = Client(clients[n], config)
                q_new_n, t_new = client.update_q(q_global_list[n], parameters)
                clients[n]["mean"] = t_new["mean"]
                clients[n]["variance"] = t_new["variance"]
                clients[n]["variance_inverse"] = t_new["variance_inverse"]
                clients[n]["iteration"] += 1
                
                #elbo_i += q_new_n["metrics"]["elbos"][i]
            
            if global_div != "KLD" and i % 1 == 0:
                prior, q_global = self.server(prior, clients, q_global, parameters, config)
            else:
                q_global = self.KL_server(prior, clients, q_global, q_global_list, config)

            prior_list.append(copy.deepcopy(prior))
            q_list.append(copy.deepcopy(q_global))
        
        print("\n", "prior distributions: \n")
        i=0
        for pi in prior_list:
            print(i, ": loc: ", pi["loc"].detach(), "var: ", pi["var"].detach(), "\n")
        print("\n","=====================================================================\n", "server approximations: \n")
        i=0
        for q in q_list:
            print(i, ": loc: ", q["loc"].detach(), "var: ", q["var"].detach(), "\n")
            
        print("Final Prior with loc: ", pi["loc"].detach(), "var: ", pi["var"].detach(), "and posterior with loc: ", q["loc"].detach(), "var: ", q["var"].detach())
        return q_global, prior, clients

    def server(self, prior, clients, q_global, parameters, config):
        #Equation 4 & 5 combined at the server with Gaussian approximations
        prior_temp, metric = self.GVI_gaussians_prior(prior, clients, q_global, config)
        #print("Temporary approximation with parameters: ", prior_temp)

        if config["is_1d"] == False: 
            print("Not 1d. Not implmeneted for other dimensions")
        #Find the new prior and the current posterior
        if clients[0]["spherical"] == False:
            print("ERROR")
            q_var = 0.
            q_loc = 0.
        else:
            if config["is_1d"]:
                
                q_var_inv = torch.exp(prior_temp["var"].detach()) ** -1
                q_loc = (torch.exp(prior_temp["var"].detach()) ** -1) * prior_temp["loc"].detach()
                for client in clients:
                    q_var_inv += client["variance_inverse"]
                    q_loc += (client["mean"] / client["variance"])
                
                q_var = q_var_inv ** -1
                q_loc *= q_var
                
            else:
                print("ERROR")
                q_var = 0.
                q_loc = 0.

        """if prior_temp["var"].detach() < 0:
            print("!!! New prior variance is less than 0!!!")
            print("Initial var: ", prior_temp["var"].detach() ** -1)
            s = 0.
            for c in clients:
                print(c["variance_inverse"])
                s += c["variance_inverse"]
            print("Clients summation over vars: ", s)"""

        """if clients[0]["spherical"] == False:
            q_var_inverse = pr_var_inverse
            q_mean_summation = torch.linalg.solve(prior["var"], prior["loc"].unsqueeze(-1))
            for client in clients:
                q_var_inverse += client["variance_inverse"]
                q_mean_summation += torch.linalg.solve(client["variance"], client["mean"].unsqueeze(-1))
            q_var = torch.linalg.inv(q_var_inverse)
            q_mean = torch.matmul(q_var, q_mean_summation).squeeze(1)
        else:
            if config["is_1d"]:
                q_var_inverse = pr_var_inverse
                q_mean_summation = pr_var_inverse * prior["loc"]
                for client in clients:
                    q_var_inverse += client["variance_inverse"]
                    q_mean_summation += client["variance_inverse"] * client["mean"]
                q_var = q_var_inverse ** -1
                q_mean = q_var * q_mean_summation
            else:
                q_var_inverse = pr_var_inverse
                q_mean_summation = torch.linalg.solve(prior["var"] * torch.eye(config["D"]), prior["loc"].unsqueeze(-1))
                for client in clients:
                    q_var_inverse += client["variance_inverse"]
                    q_mean_summation += torch.matmul(client["variance_inverse"] * torch.eye(config["D"]), client["mean"].unsqueeze(-1))
                q_var = q_var_inverse ** -1
                q_mean = torch.matmul(q_var * torch.eye(config["D"]), q_mean_summation).squeeze(1)
        
        q_global = {
            "loc": torch.nn.Parameter(q_mean),
            "var": torch.nn.Parameter(q_var),
        }"""
        q_global["metrics"]["elbos"].append(metric["elbos"])
        q_global["metrics"]["kls"].append(metric["kls"])
        q_global["metrics"]["lls"].append(metric["lls"])

        q_global.update({
            "loc": nn.Parameter(q_loc),
            "var": nn.Parameter(q_var)})
        prior = {
            "loc": nn.Parameter(prior_temp["loc"].detach()),
            "var": nn.Parameter(torch.exp(prior_temp["var"].detach())),
        }
        
        print("New Prior with parameters: ", prior)
        print("New approximation with parameters: ", q_global["loc"], q_global["var"])
        
        return prior, q_global
        
    def GVI_gaussians_prior(self, prior_og, clients, q_global, config):

        print("Server approximation step")
        print("Prior optimisation with parameters: ")
        # We initialise our approximation as the previous approximation
        #l = torch.tensor([0.0]) #
        l = torch.tensor([0.])
        print("prior mean: ", l, "initial mu")
        
        v = torch.tensor([1.])
        print("prior variance: ",v, " which is fixed as v of new dist")
        pi_s = {
            "loc": torch.nn.Parameter(l),
            "var": torch.nn.Parameter(torch.log(v)),
        }
        prior = {
            "loc": prior_og["loc"].detach(),
            "var": prior_og["var"].detach()
        }
        print("PRIOR: ", prior)
            
        q_params = list(pi_s.items())
            
        q_parameters = [
                        {"params": pi_s["loc"]},
                        {"params": pi_s["var"]}
                    ]    
        
        #print(q_parameters)
        optimiser_s = optim.SGD(q_parameters, lr=config["lr"])
        print("Optimiser: ", optimiser_s)
        
        lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            optimiser_s, **config["lr_scheduler_params"])

        metric = {
            "elbos": [],
            "kls": [],
            "lls": []
        }

        if clients[0]["spherical"]:
            v_loss_ = 0.0
            mu_loss = 0.0
            
            for client in clients:
                assert torch.abs((client["variance"] ** -1) -client["variance_inverse"]) < 1e-10, f'Inverse of Client variance and client variance_inverse differ!, difference: {(client["variance"] ** -1) - client["variance_inverse"]}'
                v_loss_ += client["variance_inverse"]
                mu_loss += (client["mean"]) / client["variance"]
            
            v_loss = v_loss_ ** -1
            mu_loss = v_loss * mu_loss
        else:
            v_loss = 0.0
            mu_loss = 0.0
        print("Divergence: ", config["global_div"], "with parameter: ", config["div_hyper_param"])
        print("Server loss with parameters: ", mu_loss, v_loss)

        for i in range(config["optim_epochs"]):
            
            optimiser_s.zero_grad()
            
            
            if clients[0]["spherical"]:
                var = (v_loss * torch.exp(pi_s["var"])) / (v_loss + torch.exp(pi_s["var"]))
                mu = (mu_loss * torch.exp(pi_s["var"]) + pi_s["loc"] * v_loss) / (v_loss + torch.exp(pi_s["var"]))

                if config["global_div"] == "AR":
                    div = Divergences().alpha_renyi_spherical(mu, prior["loc"].detach(), var, prior["var"].detach(), config, config["div_hyper_param"])
                elif config["global_div"] == "RKL":
                    div = Divergences().reverse_kl_spherical(mu, prior["loc"].detach(), var, prior["var"].detach(), config)
                elif config["global_div"] == "wKL":
                    div = Divergences().kl_spherical_gaussian(mu, prior["loc"].detach(),var, prior["var"].detach(), config) / config["div_hyper_param"]
                elif config["global_div"] == "KLD":
                    div = Divergences().kl_spherical_gaussian(mu, prior["loc"].detach(),var, prior["var"].detach(), config)
                elif config["global_div"] == "G":
                    div = Divergences().gamma_spherical(mu, prior["loc"].detach(), var, prior["var"].detach(), config, config["div_hyper_param"])
                elif config["global_div"] == "FR":
                    if config["is_1d"]:
                        div = Divergences().FisherRao_normals(mu, prior["loc"].detach(), var, prior["var"].detach())
                    else:
                        print("!The Fisher Rao divergence is not implemented for multivariate Gaussian distributions as it has no closed form!")
                else:
                    print("!!! No Divergence specified or NOT Yet Implemented!!!\n The Arguement specified is: ", config["global_div"])
                    div = 0
                #assert div > 0, "Negative Divergence encountered????"
            else:
                var = torch.linalg.inv(torch.linalg.inv(torch.exp(pi_s["var"])) + torch.linalg.inv(v_loss))
                mu = var @ (torch.matmul(torch.exp(pi_s["var"]), pi_s["loc"].unsqueeze(-1)) + torch.matmul(v_loss, mu_loss))
                
                if config["global_div"] == "AR":
                    div = Divergences().alpha_renyi(q_s["loc"], prior["loc"], q_s["var"], prior["var"], config["div_hyper_param"])
                elif config["global_div"] == "RKL":
                    div = Divergences().reverse_kl(q_s["loc"], prior["loc"], q_s["var"], prior["var"])
                elif config["global_div"] == "wKL":
                    div = Divergences().kl_gaussians(q_s["loc"], prior["loc"], q_s["var"], prior["var"]) / config["div_hyper_param"]
                else:
                    print("!!! No Divergence specified or NOT Yet Implemented!!!\n The Arguement specified is: ", config["global_div"])
                    div = torch.tensor([0.])

            assert config["is_1d"], "Not univariate Gaussian encountered. Currently not implemented for multivariate distributions."
            ell = v_loss * (((pi_s["loc"] - mu_loss) / (v_loss + torch.exp(pi_s["var"]))) ** 2) + torch.exp(pi_s["var"]) / (torch.exp(pi_s["var"]) + v_loss)
 
            """for client in clients:
                dif = mu - client["mean"]
                if client["spherical"]:
                    if config["is_1d"]:
                        ell += (dif ** 2) / (client["variance"])
                        ell +=  var / (client["variance"])
                    else:
                        ell += torch.matmul(dif.unsqueeze(0), torch.matmul(client["variance_inverse"] * torch.eye(config["D"]), dif.unsqueeze(-1))).squeeze() 
                        ell += torch.matmul(client["variance_inverse"] * torch.eye(config["D"]), var * torch.eye(config["D"]))
                else:
                    ell += torch.matmul(dif.unsqueeze(0), torch.matmul(client["variance_inverse"], dif.unsqueeze(-1))).squeeze() 
                    ell += torch.trace(torch.matmul(client["variance_inverse"], var))"""
                
            ell /= 2

            loss = ell + div

            #print(ell, div, loss)
            
            loss.backward()
            #print(i, "Current value: ", pi_s["loc"], "Gradient: ", pi_s["loc"].grad, "lr", optimiser_s.param_groups[0]['lr'], "Should be update: ", pi_s["loc"] - optimiser_s.param_groups[0]['lr'] * pi_s["loc"].grad)
            optimiser_s.step()
            #print("PRINTING VALS", pi_s["var"], "loss: ", mu_loss, v_loss, "prior: ", prior["loc"],  prior["var"])
            if i % 10 == 0:
                metric["elbos"].append(loss.item())
                metric["kls"].append(div.item())
                metric["lls"].append(ell.item())
        
        lr_scheduler.step()
        print(optimiser_s)
        return pi_s, metric

    # Not yet applicable to the Spherical setting
    def KL_server(self, prior, clients, q_global, q_global_list, config):
        if config["is_1d"]:
            var_s_inv = prior["var"].detach() ** -1
            mu_s = prior["loc"].detach() / prior["var"].detach()
            
            for client in clients:
                var_s_inv += client["variance_inverse"]
                mu_s += client["mean"] / client["variance"]
                
            var_s = var_s_inv ** -1
            loc_s = var_s * mu_s

            q_new = copy.deepcopy(q_global)
            q_new.update({"loc": nn.Parameter(loc_s), "var": nn.Parameter(var_s)})
            
        else:
            q_new = self.KL_server_outdated(q_global, q_global_list, config)

        print("Server Update \n", "New approximation at the server: with parameters loc:", q_new["loc"], "and var: ", q_new["var"])
        return q_new

    def KL_server_outdated(self, q_global, q_global_list, config):    
        M = len(q_global_list)
        print(q_global_list)
        if config["is_1d"]:
            print("!ERROR! \n Should not access 'KL_server_outdated' if 1-dimensional!")
        else:
            denom_var_inv = (M-1) * torch.cholesky_solve(torch.linalg.cholesky(q_global["var"].detach()+ JITTER*torch.eye(config["D"])), torch.eye(config["D"]))
            denom_var = torch.cholesky_solve(torch.linalg.cholesky(denom_var_inv + JITTER*torch.eye(config["D"])), torch.eye(config["D"]))
            
            denom_mean = torch.matmul(denom_var, (M-1) * torch.linalg.solve(
                q_global["var"].detach(), q_global["loc"].detach().unsqueeze(-1)))
            
            numer_var_inv = torch.zeros_like(q_global["var"].detach())
            for q in q_global_list:
                numer_var_inv += torch.linalg.inv(q["var"].detach())
                
            numer_var = torch.linalg.inv(numer_var_inv)
            
            numer_var_mean = torch.linalg.solve(
                q_global_list[0]["var"].detach(), q_global_list[0]["loc"].detach().unsqueeze(-1))
            
            for i in range(1, len(q_global_list)):
                numer_var_mean += torch.linalg.solve(
                    q_global_list[i]["var"].detach(), q_global_list[i]["loc"].detach().unsqueeze(-1))
                
            numer_mean = torch.matmul(numer_var, numer_var_mean)
            
            q_new_var_inv = numer_var_inv - denom_var_inv
            
            q_new_var = torch.linalg.inv(q_new_var_inv)
            
            q_new_var_mean = torch.matmul(numer_var_inv, numer_mean) - torch.matmul(denom_var_inv, denom_mean)
            
            q_new_mean = torch.matmul(q_new_var, q_new_var_mean).squeeze(-1)
            
            q_new = copy.deepcopy(q_global)
            
        q_new.update({"loc": nn.Parameter(q_new_mean), "var": nn.Parameter(q_new_var)})
        
        return q_new