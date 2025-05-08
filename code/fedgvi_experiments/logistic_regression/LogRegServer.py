import numpy as np
import torch
import torch.nn as nn
import copy
from tqdm import tqdm

from fedgvi_experiments.logistic_regression.LogRegClient import LogRegClient
from fedgvi_experiments.utils.helper_functions import helper_functions
from fedgvi_experiments.utils.Divergences import Divergences


torch.set_default_dtype(torch.float64)
# =============================================================================
# Logistic Regression Server
# =============================================================================

# Synchronous implementation for FedGVI on Logistic Regression with Gaussian approximation on the weights
# All covariance matrices are assumed to be diagonal so that the weight vector becomes a vector of Gaussian r.v.s

class LogRegServer:

    def help(self):
        return {
            "Client": LogRegClient().help(), 
            "Server": self.server_help()
        }
    
    def server_help(self):
        return {
            "D": "Dimension of data",
            "N": 'Number of data points in entire training data set',
            "Epochs": "Number of client iterations",
            "global_div": "Server divergence, options are: 'KLD'",
            "global_div_param": "Server divergence hyper-parameter",
            "lr": "Torch optimiser learning rate, default '5e-3'",
            "num_samples": "If using Monte Carlo approximation, the number of samples drawn",
            "optim_epochs": "Number of optimisation Epochs",
            "batch_size": "If using minibatches, this is the size of them",
            "default_seed": "Seed for RNG in Monte Carlo sampling"
        }

    def FedGVI(self, q_global, clients, parameters, test_data, minibatch=False, batched_clients=1.0):
        
        D = parameters["D"]
        
        global_div = parameters["global_div"]
        div_hyper_param = parameters["global_div_param"]
        
        if minibatch:
            batchsize = parameters["batch_size"]
        else:
            batchsize = np.inf
            
        mu_pi = copy.deepcopy(q_global["loc"].detach())
        v_pi = copy.deepcopy(q_global["var"].detach())
        
        prior = {
            "loc": mu_pi,
            "var": v_pi,
        }
        print("\nHello there\n")
        config = {
            "D": D,
            "epochs": parameters["Epochs"], 
            "num_samples": parameters["num_samples"],
            "lr_scheduler_params": {"lr_lambda": lambda epoch: 1.0},
            "optim_epochs": parameters["optim_epochs"],
            "lr": parameters["lr"],
            "early_stopping": False,
            "minibatch": minibatch,
            "batchsize": batchsize,
            "global_div": global_div,
            "div_hyper_param": div_hyper_param,
            "default_seed": parameters["default_seed"],
        }
        
        #print("Starting global q: ", q_global)
        
        q_list = [copy.deepcopy(prior)]
        test_val = []

        for i in range(config["epochs"]):
            elbo_i = 0.
            
            q_global_list = []
            for n in range(len(clients)):
                temp = copy.deepcopy(q_global)
                q_global_list.append(temp)
            
            if batched_clients < 1.0: #Random subset of clients chosen represented as fraction of clients in batch
                batch_indices = helper_functions.minibatch_of_indices(batched_clients, len(clients), seed=(config["default_seed"] + i))
                for n in tqdm (range(len(clients)), desc=f"Completed Clients at Iteration {i+1}", colour='blue'):
                    if n in batch_indices:
                        #print(f"Client {n}")
                        client = LogRegClient(clients[n], config)
                        q_new_n, t_new = client.update_q(q_global_list[n], parameters)
                        clients[n]["mean"] = t_new["mean"]
                        clients[n]["variance"] = t_new["variance"]
                        clients[n]["variance_inverse"] = t_new["variance_inverse"]
                        clients[n]["iteration"] += 1
                    else:
                        #print(f"Client {n}")
                        clients[n]["iteration"] += 1
            else: #Update all clients
                for n in tqdm (range(len(clients)), desc=f"Completed Clients at Iteration {i+1}", colour='blue'):
                    #print(f"Client {n}")
                    client = LogRegClient(clients[n], config)
                    q_new_n, t_new = client.update_q(q_global_list[n], parameters)
                    clients[n]["mean"] = t_new["mean"]
                    clients[n]["variance"] = t_new["variance"]
                    clients[n]["variance_inverse"] = t_new["variance_inverse"]
                    clients[n]["iteration"] += 1
                
            # We currently only assume that the server uses a Kullback-Leibler divergence
            """if global_div != "KLD":
                prior, q_global = self.server(prior, clients, q_global, parameters, config)
            else:"""
            q_global = self.KL_server(prior, clients, q_global)
            #q_global = self.Optim_server(prior, clients, q_global, config)
            #print("q_new", q_global)
            q_list.append(copy.deepcopy(q_global))

            acc = self.Log_reg_validate_accuracy(q_global, test_data)

            print("Validation accuracy: ", acc)

            test_val.append(acc)
        
        return {
            "q": q_global, 
            "clients": clients, 
            "validation": test_val, 
            "q_list": q_list
            }

    def Log_reg_validate_accuracy(self, q, test_data):
        # Using the Probit approximation of Spiegelhalter and Lauritzen (1990) as given in Ashman et al. (2022)
        # We assume {0,1} labels
        acc = 0.0
        mu = q["loc"]
        
        sigma = q["var"]
        for i in tqdm (range(len(test_data["y"])), desc="Validating", colour='green'):

            assert test_data["y"][i] == 0 or test_data["y"][i] == 1, "Not {0,1} loss labels."

            x = test_data["x"][i]
            z = (mu @ x) / torch.sqrt(1 + np.pi * (x @ torch.diag(sigma) @ x))
            apx_sigmoid = torch.special.expit(z)

            if apx_sigmoid > 0.5:
                apx = 1
            else:
                apx = 0
            if apx == test_data["y"][i]:
                acc += 1

        acc /= len(test_data["y"])
        
        return acc
    
    def KL_server(self, prior, clients, q_global):
        # We assume that we have either diagonal covariance matrices or a scaled identity matrix 
        #print("=============================================")
        #print("Prior:", prior)
        #print("=============================================")
        mu = copy.deepcopy(prior["loc"])
        sigma_inv = copy.deepcopy(prior["var"]) ** -1 # Vector of the diagonal on covariance matrix

        for client in clients:
            sigma_inv += client["variance"] ** -1 # variance vector representing the diagonal of covariance matrix

            mu += (client["variance"] ** -1) * client["mean"] # elementwise multiplication of two vectors
        
        cov = sigma_inv ** -1
        loc = cov * mu

        #print("Cov at server:", cov)

        q_new = copy.deepcopy(q_global)
        q_new.update({
            "loc": nn.Parameter(loc),
            "var": nn.Parameter(cov)
        })

        return q_new
    
    def Optim_server(self, prior, clients, q, config):
        
        combined_loss = self.combine_losses(clients, config)

        q_s = copy.deepcopy(q)

        prev_var = copy.deepcopy(q["var"].detach())
        mu = copy.deepcopy(q["loc"].detach())
        v = torch.log(prev_var)

        q_s.update({
            "loc": torch.nn.Parameter(mu),
            "var": torch.nn.Parameter(v),
        })
        #print("q_s: ", q_s)

        q_parameters = [
                        {"params": q_s["loc"]},
                        {"params": q_s["var"]}
                    ]    
        
        optimiser = torch.optim.Adam(q_parameters, lr=config["lr"])
        
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimiser, **config["lr_scheduler_params"])
        
        for _ in tqdm (range(5*config["optim_epochs"]), desc=f"Server Optimisation", colour='red'):
            optimiser.zero_grad()

            div = 0.0
            div = self.compute_divergence(q_s, prior, config["global_div"], config["div_hyper_param"])

            ll = 0.0
            # Check cookbook for direct implementation of this. Is this possible with non PSD covariance matrices inside the expectation?

            q_dist = torch.distributions.MultivariateNormal(loc=q_s["loc"], covariance_matrix=torch.diag(torch.exp(q_s["var"])))
            thetas = q_dist.rsample((config["num_samples"],)) # rsample allows differentation through the expectation

            ll += ((thetas - combined_loss["loc"]) * (combined_loss["var"] ** -1) * (thetas - combined_loss["loc"])).sum()

            loss = div + ll

            loss.backward()
            optimiser.step()

        mu_new = q_s["loc"].detach()
        v_new = torch.exp(copy.deepcopy(q_s["var"].detach()))

        q_s.update({
            "loc": mu_new,
            "var": v_new
        })

        lr_scheduler.step()
        #print("\nRet q_s: ", q_s)
        return q_s

    def combine_losses(self, clients, config):
        
        mu = 0.0
        sigma_inv = 0.0 # Vector of the diagonal on covariance matrix

        for client in clients:
            sigma_inv += client["variance"] ** -1 # variance vector representing the diagonal of covariance matrix

            mu += (client["variance"] ** -1) * client["mean"] # elementwise multiplication of two vectors
        
        cov = sigma_inv ** -1
        loc = cov * mu

        return {
            "loc": loc,
            "var": cov,
        }
    
    def compute_divergence(self, q, prior, div_name, param=None):
        mean = q["loc"]
        cov = torch.diag(torch.exp(q["var"]))

        divergences = ["KLD", "RKL", "wKL", "AR"]
        assert div_name in divergences, f"Please choose a valid divergence from: {divergences}"

        if div_name == "KLD":
            div =  Divergences().kl_gaussians(mean, prior["loc"], cov, torch.diag(prior["var"]))
        elif div_name == "RKL":
            div =  Divergences().reverse_kl(mean, prior["loc"], cov, torch.diag(prior["var"]))
        elif div_name == "wKL":
            div =  Divergences().kl_gaussians(mean, prior["loc"], cov, torch.diag(prior["var"])) / param
        elif div_name == "AR":
            div =  Divergences().alpha_renyi(mean, prior["loc"], cov, torch.diag(prior["var"]), param)
        else:
            div = 0.0
        
        return div