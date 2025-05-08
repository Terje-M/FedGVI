import numpy as np
import torch
from torch import optim
import copy
import math

from fedgvi_experiments.utils.Divergences import Divergences
from fedgvi_experiments.utils.helper_functions import helper_functions

device = "cpu"
torch.set_default_dtype(torch.float64)
# =============================================================================
# Logistic Regression Client
# =============================================================================

class LogRegClient:

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

    def get_default_config(self):
        return {
            "epochs": 200,
            "num_samples": 100, # Number of Monte-Carlo samples
            "optimiser": "Adam",
            "optimiser_params": {"lr": 0.05},
            "model_optimiser_params": {},
            "sigma_optimiser_params": {},
            "lr_scheduler": "MultiplicativeLR",
            "lr_scheduler_params": {"lr_lambda": lambda epoch: 1.0},
            "minibatch": False,
            "batch_size": np.inf,
            "default_seed": 42,
        }
    
    def get_default_client(self):
        return {
            "client_idx": 0,
            "data": {"x": [], "y": []},
            "mean": None,
            "variance": None,
            "variance_inverse": None,
            "iteration": 0,
            "Divergence": "KLD",
            "div_param": None,
            "loss": "nll",
            "score_function": None,
            "loss_param": None,
        }

    def update_q(self, q, parameters):
        q_new, t_new = self.gradient_based_update(q, parameters)
            
        return q_new, t_new
    
    def gradient_based_update(self, q, parameters):
        q_old = copy.deepcopy(q)
        q_s = copy.deepcopy(q)
        q_cav = self.cavity(q)
        #print("Cavity: ", q_cav)
        #print("q_old", q_old)
        #print("Client", self.client["mean"], self.client["variance"])
        #print("q_cav", q_cav)
        prev_var = copy.deepcopy(q["var"].detach())
        mu = copy.deepcopy(q["loc"].detach())
        v = torch.log(prev_var)

        q_s.update({
            "loc": torch.nn.Parameter(mu),
            "var": torch.nn.Parameter(v),
        })
        #print("client q_m initial: ", q_s)
        
        q_parameters = [
                        {"params": q_s["loc"]},
                        {"params": q_s["var"]}
                    ]    
        
        optimiser = optim.Adam(q_parameters, lr=parameters["lr"])
        
        lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            optimiser, **self.config["lr_scheduler_params"])
        
        for n in range(self.config["optim_epochs"]):
            
            optimiser.zero_grad()
            
            coef = 1.0
            batch = self.client["data"]
            if self.config["minibatch"]:
                coef = len(self.client["data"]["y"]) / self.config["batchsize"]
                seed_n = self.config["default_seed"] + self.client["iteration"] * self.config["optim_epochs"] + n # Allows different subsets but keeps reproducability 
                batch = helper_functions.get_xy_batch(self.client["data"], self.config["batchsize"], seed=seed_n)

            div = 0.0
            div = self.compute_divergence(q_s, q_cav)

            ll = 0.0

            for e in q_s["var"]:
                if math.isnan(e):
                    print(q_s["var"])
                    break

            if self.client["loss"] not in ["gen_CE", "gamma_mislabel", "density_power","nll"]:
                if n == 0:
                    print("Specified loss is not one of: 'gen_CE', 'gamma_mislabel', or 'nll', but rather: ", self.client["loss"])
            if self.client["loss"] == "gen_CE":
                # Using the Generalized Cross Entropy Loss of Zhang and Sabuncu (2018)
                assert self.client["loss_param"] > 0 and self.client["loss_param"] <= 1, " Need the loss parameter b ∊ (0,1]"
                q_dist = torch.distributions.MultivariateNormal(loc=q_s["loc"], covariance_matrix=torch.diag(torch.exp(q_s["var"])))
                thetas = q_dist.rsample((self.config["num_samples"],)) # rsample allows differentation through the expectation

                X = batch["x"]
                Y = batch["y"]
                
                # Positive Log Likelihood explicit formula as in nll case
                pll = ((Y * X.T).T @ thetas.T)
                pll -= (1 + torch.exp(X @ thetas.T)).log()

                gen_ce = (len(self.client["data"]["y"]) - coef * torch.exp(self.client["loss_param"] * pll).sum()) / self.client["loss_param"]
                
                ll = - gen_ce
                
            elif self.client["loss"] == "gamma_mislabel":
                # Using the Gamma Divergence based loss of Hung et al. (2018)

                q_dist = torch.distributions.MultivariateNormal(loc=q_s["loc"], covariance_matrix=torch.diag(torch.exp(q_s["var"])))
                thetas = q_dist.rsample((self.config["num_samples"],)) # rsample allows differentation through the expectation

                X = batch["x"]
                Y = batch["y"]
                g = self.client["loss_param"]

                part1 = torch.exp(Y * (g + 1) * (thetas @ X.T))
                part2 = 1 + torch.exp((g + 1) * thetas @ X.T)
                gamma_mis = torch.pow(part1 / part2, g / (g + 1)).sum()
                
                gamma_mis /= self.config["num_samples"]

                ll = gamma_mis

            elif self.client["loss"] == "density_power":
                # Using the Density Power Divergence based loss of Gosh and Basu (2016)
                q_dist = torch.distributions.MultivariateNormal(loc=q_s["loc"], covariance_matrix=torch.diag(torch.exp(q_s["var"])))
                thetas = q_dist.rsample((self.config["num_samples"],)) # rsample allows differentation through the expectation

                X = batch["x"]
                Y = batch["y"]
                b = self.client["loss_param"]

                dpd = ((1 + torch.exp((1 + b) * thetas @ X.T)) / torch.pow(1 + torch.exp(thetas @ X.T), 1 + b)).sum() \
                        - ((1+b) * torch.exp(((b * Y * X.T).T @ thetas.T).T - (b * ((1 + torch.exp(thetas @ X.T)).log()))).sum() / b)
                
                dpd /= self.config["num_samples"]
                
                ll = - dpd

            else: # Default logistic loss with Log Bernoulli likelihood, we use the exponential family formulation of a Bernoulli
                # For a derivation of the expression below see Murphy (2023) Eqn. (15.134)
                # Use reparametrisation trick implicitly (as defined through PyTorch) to do Monte Carlo integration
                
                q_dist = torch.distributions.MultivariateNormal(loc=q_s["loc"], covariance_matrix=torch.diag(torch.exp(q_s["var"])))
                thetas = q_dist.rsample((self.config["num_samples"],)) # rsample allows differentation through the expectation

                X = batch["x"]
                Y = batch["y"]
                
                ll += ((Y * X.T).T @ thetas.T).sum()
                ll -= (1 + torch.exp(X @ thetas.T)).log().sum()
                ll /= self.config["num_samples"]

            ll *= coef

            loss = div - ll

            loss.backward()
            optimiser.step()

        mu_new = q_s["loc"].detach()
        v_new = torch.exp(copy.deepcopy(q_s["var"].detach()))

        q_s.update({
            "loc": mu_new,
            "var": v_new
        })
        #print("client q_m ret: ", q_s)
        t_new = self.update_client_t(q_s, q_old)
        lr_scheduler.step()

        #print("q_m", q_s)
        #print("t_new", t_new)
        return q_s, t_new
    
    def compute_divergence(self, q, q_cav):
        mean = q["loc"]
        cov = torch.diag(torch.exp(q["var"]))

        divergences = ["KLD", "RKL", "wKL", "AR"]
        assert self.client["Divergence"] in divergences, f"Please choose a valid divergence from: {divergences}"

        if self.client["Divergence"] == "KLD":
            div =  Divergences().kl_gaussians(mean, q_cav["loc"], cov, torch.diag(q_cav["var"]))
        elif self.client["Divergence"] == "RKL":
            div =  Divergences().reverse_kl(mean, q_cav["loc"], cov, torch.diag(q_cav["var"]))
        elif self.client["Divergence"] == "wKL":
            div =  Divergences().kl_gaussians(mean, q_cav["loc"], cov, torch.diag(q_cav["var"])) / self.client["div_param"]
        elif self.client["Divergence"] == "AR":
            div =  Divergences().alpha_renyi(mean, q_cav["loc"], cov, torch.diag(q_cav["var"]), self.client["div_param"])
        else:
            div = 0.0
        
        return div
    
    def cavity(self, q):
        # Diagonal Covariance matrix is parametrised as a single vector and passed as diagonal matrix in computation
        if self.client["iteration"] == 0:
            v_bar_n = q["var"].detach()
            m_bar_n = q["loc"].detach() 
        else:
            v_bar_n_inverse = (q["var"].detach() ** -1) - (self.client["variance"] ** -1)
            v_bar_n = v_bar_n_inverse ** -1

            part = (q["var"].detach() ** -1) * q["loc"].detach() 
            part -= (self.client["variance"] ** -1) * self.client["mean"]
            m_bar_n = (v_bar_n) * part

        return {
            "loc": m_bar_n,
            "var": v_bar_n
        }
    
    def update_client_t(self, q, q_old):
        if self.client["iteration"] == 0:
            variance_inverse = (q["var"].detach() ** -1) - (q_old["var"].detach() ** -1)
            variance = variance_inverse ** -1

            mean_ = (torch.diag((q["var"].detach() ** -1)) @ q["loc"].detach()) 
            mean_ -= (torch.diag((q_old["var"].detach() ** -1)) @ q_old["loc"].detach())

            mean = torch.diag(variance) @ mean_
        else:
            old_var = self.client["variance"]
            old_mean = self.client["mean"]

            variance_inverse = (q["var"].detach() ** -1) - (q_old["var"].detach() ** -1) + (old_var ** -1)
            variance = variance_inverse ** -1

            mean_ = ((q["var"].detach() ** -1) * q["loc"].detach()) 
            mean_ -= ((q_old["var"].detach() ** -1) * q_old["loc"].detach())
            mean_ +=  ((old_var ** -1) * old_mean)

            mean = torch.diag(variance) @ mean_

        return {
            "mean": mean,
            "variance": variance,
            "variance_inverse": variance_inverse,
        }