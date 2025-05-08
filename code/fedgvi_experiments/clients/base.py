import logging
import torch
import numpy as np
import math

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
#from tqdm.auto import tqdm

from fedgvi_experiments.utils.training_utils import EarlyStopping
from fedgvi_experiments.utils.training_utils import Timer

logger = logging.getLogger(__name__)


# =============================================================================
# Client class 
# =============================================================================

#update_time
class Client:
    def __init__(self, data, model, t=None, config=None, val_data=None):
        
        if config is None:
            config = {}

        self._config = self.get_default_config()
        self.config = config

        # Set data partition and likelihood
        self.data = data
        
        self.model = model

        # Set likelihood approximating term
        self.t = t

        # Maintain optimised approximate posterior.
        self.q = None

        # Validation dataset.
        self.val_data = val_data

        self.log = defaultdict(list)
        self._can_update = True

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = {**self._config, **config}

    def get_default_config(self):
        return {
            "train_model": False,
            "damping_factor": 1.0,
            "valid_factors": False,
            "update_log_coeff": False,
            "epochs": 1,
            "batch_size": 100,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 0.05},
            "model_optimiser_params": {},
            "sigma_optimiser_params": {},
            "lr_scheduler": "MultiplicativeLR",
            "lr_scheduler_params": {"lr_lambda": lambda epoch: 1.0},
            "early_stopping": EarlyStopping(np.inf),
            "performance_metrics": None,
            "track_q": False,
            "num_elbo_samples": 10,
            "print_epochs": np.pi,
            "device": "cpu",
            "verbose": False,
            "no_step_first_epoch": False,
            "log_training": True,
            "log_performance": True,
            "divergence": "KLD",
            "alpha": 0.5,
            "loss": "nll",
            "loss_param": 1.0,
        }

    def can_update(self):
        """
        A check to see if this client can indeed update. Examples of reasons
        one may not be is that they haven't finished optimisation.
        """
        return self._can_update

    def evaluate_performance(self, default_metrics=None):
        metrics = {}
        if default_metrics is not None:
            metrics = {**default_metrics, **metrics}

        if self.config["performance_metrics"] is not None:
            train_metrics = self.config["performance_metrics"](self, self.data)
            for k, v in train_metrics.items():
                metrics["train_" + k] = v

            if self.val_data is not None:
                val_metrics = self.config["performance_metrics"](self, self.val_data)
                for k, v in val_metrics.items():
                    metrics["val_" + k] = v

        return metrics

    def fit(self, *args, **kwargs):
        """
        Computes the refined approximating posterior (q) and associated
        approximating likelihood term (t). This method differs from client to
        client, but in all cases it calls Client.q_update internally.
        """
        return self.update_q(*args, **kwargs)

    def update_q(self, q, init_q=None, **kwargs):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """

        timer = Timer()
        timer.start()

        # Type(q) is self.model.conjugate_family. 
        if (
            str(type(q)) == str(self.model.conjugate_family)
            and not self.config["train_model"]
        ):
            # No need to make q trainable.
            self.q, self.t = self.model.conjugate_update(self.data, q, self.t)
        else:
            # Pass a trainable copy to optimise.
            self.q, self.t = self.gradient_based_update(p=q, init_q=init_q, **kwargs)

        times = timer.get()
        self.log["update_time"].append(times)

        return self.q, self.t

    def gradient_based_update(self, p, init_q=None, **kwargs):
        # Cannot update during optimisation.
        self._can_update = False

        # Copy the approximate posterior, make non-trainable.
        q_old = p.non_trainable_copy()
        q_cav = p.non_trainable_copy()

        if self.t is not None:
            # TODO: check if valid distribution.
            q_cav.nat_params = {
                k: v - self.t.nat_params[k] for k, v in q_cav.nat_params.items()
            }

        if init_q is not None:
            q = init_q.trainable_copy()
        else:
            # Initialise to prior.
            q = p.trainable_copy()

        q_parameters = list(q.parameters())
        if self.config["train_model"]:
            parameters = [
                {"params": q_parameters[0]},
                {"params": q_parameters[1], **self.config["sigma_optimiser_params"]},
                {
                    "params": self.model.parameters(),
                    **self.config["model_optimiser_params"],
                },
            ]
        else: 
            parameters = [
                {"params": q_parameters[0]},
                {"params": q_parameters[1], **self.config["sigma_optimiser_params"]},
            ]

        # Reset optimiser.
        logging.info("Resetting optimiser")
        if self.config["optimiser"] == "sgdm":
            optimiser = torch.optim.SGD(
                parameters, **self.config["optimiser_params"], momentum=0.9
            )
        else:
            optimiser = getattr(torch.optim, self.config["optimiser"])(
                parameters, **self.config["optimiser_params"]
            )
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.config["lr_scheduler"])(
            optimiser, **self.config["lr_scheduler_params"]
        )
        #print(optimiser)
        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(
            tensor_dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        if "cuda" in self.config["device"]:
            loader.pin_memory = True

        # Dict for logging optimisation progress.
        training_metrics = defaultdict(list)

        # Dict for logging performance progress.
        performance_metrics = defaultdict(list)

        # Reset early stopping.
        self.config["early_stopping"](scores=None, model=q.non_trainable_copy())

        # Gradient-based optimisation loop -- loop over epochs.
        epoch_iter = tqdm(
            range(self.config["epochs"]),
            desc="Epoch",
            leave=True,
            disable=(not self.config["verbose"]),
            position=0,
        )
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = defaultdict(lambda: 0.0)
            epoch["elbos"] = []
            epoch["divs"] = []
            epoch["lls"] = []


            stopped = False
            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                x_batch = x_batch.to(self.config["device"])
                y_batch = y_batch.to(self.config["device"])

                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }
                
                #Determine the divergence used and compute this term
                if self.config["divergence"] == "KLD":
                    # Compute the KL divergence between q and q_cav, ignoring
                    # A(η_cav).
                    #if i == 0:
                        #print("KL divergence used")
                    div = q.kl_divergence(q_cav, calc_log_ap=False).sum() / len(x)
                elif self.config["divergence"] == "RKL":
                    
                    div = q_cav.kl_divergence(q, calc_log_ap=True).sum() / len(x)
                    
                elif self.config["divergence"] == "wKL":
                    div = q.kl_divergence(q_cav, calc_log_ap=False).sum() / len(x)
                    div /= self.config["alpha"]
                elif self.config["divergence"] == "AR":
                    # Compute Alpha-Rényi divergence between q and q_cav, ignoring
                    # A(η_cav).
                    #if i == 0:
                        #print("Alpha-Renyi divergence used")
                    div = q.ar_divergence(q_cav, alpha=self.config["alpha"], calc_log_ap=False).sum() / len(x)
                else:
                    print("No Divergence specified")
                    div = 0.
                
                if "loss" not in self.config.keys():
                    self.config["loss"] = self.get_default_config["loss"]
                    self.config["loss_param"] = self.get_default_config["loss_param"]

                # Sample θ from q and compute p(y | θ, x) for each θ
                if self.config["loss"] == "beta":
                    ll = self.model.expected_beta_loss(
                        batch, q, self.config["num_elbo_samples"], beta=self.config["loss_param"]
                    ).sum()
                elif self.config["loss"] == "rcce" or self.config["loss"] == "gce":
                    ll = self.model.robust_cross_entropy(
                        batch, q, self.config["num_elbo_samples"], beta=self.config["loss_param"]
                    ).sum()
                elif self.config["loss"] == "nll":
                    ll = - self.model.expected_log_likelihood(
                        batch, q, self.config["num_elbo_samples"]
                    ).sum()
                else:  # Standard log likelihood
                    print("Loss ", {self.config["loss"]}, " not recognised. Defaulting to negative log likelihood.")
                    ll = - self.model.expected_log_likelihood(
                        batch, q, self.config["num_elbo_samples"]
                    ).sum()
                ll /= len(x_batch)

                # Does the loss need to be scaled or is it an average?
                #ll *= len(y)

                #logt = torch.tensor(0.0).to(self.config["device"])

                if math.isnan(div):
                    div = 0.
                else:
                    div = div

                loss = div + ll 

       
                # Only perform gradient steps after 1st epoch.
                if self.config["no_step_first_epoch"]:
                    if i > 0:
                        loss.backward()
                        optimiser.step()
                else:
                    loss.backward()
                    optimiser.step()

                # Keep track of quantities for current batch.
                epoch["elbo"] += -loss.item() / len(loader)
                if div != 0:
                    epoch["div"] += div.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)
               

                epoch["elbos"].append(loss.item())
                if div != 0:
                    epoch["divs"].append(div.item())
                epoch["lls"].append(ll.item())


            epoch_iter.set_postfix(
                elbo=epoch["elbo"],
                div=epoch["div"],
                ll=epoch["ll"],
                lr=optimiser.param_groups[0]["lr"],
            )

            
            # Log progress for current epoch
            training_metrics["elbo"].append(epoch["elbo"])
            training_metrics["div"].append(epoch["div"])
            training_metrics["ll"].append(epoch["ll"])

            #if self.t is not None:
                #training_metrics["logt"].append(epoch["logt"])
            
            if i > 0 and i % self.config["print_epochs"] == 0:
                # Update global posterior before evaluating performance.
                self.q = q.non_trainable_copy()

                metrics = self.evaluate_performance(
                    {
                        "epochs": i,
                        "elbo": epoch["elbo"],
                        "div": epoch["div"],
                        "ll": epoch["ll"],
                    }
                )

                # Report performance.
                report = ""
                report += f"epochs: {metrics['epochs']} "
                report += f"elbo: {metrics['elbo']:.3f} "
                report += f"ll: {metrics['ll']:.3f} "
                report += f"div: {metrics['div']:.3f} \n"
                for k, v in metrics.items():
                    performance_metrics[k].append(v)
                    if "mll" in k or "acc" in k:
                        report += f"{k}: {v:.3f} "

                tqdm.write(report)
            if stopped:
                break
            # Update learning rate.
            # Only update after 1st epoch.
            if self.config["no_step_first_epoch"]:
                if i > 0:
                    lr_scheduler.step()
            else:
                lr_scheduler.step()

            # Check whether to stop early.
            if self.config["early_stopping"](
                scores=training_metrics, model=q.non_trainable_copy()
            ):
                self.q = q.non_trainable_copy()

                metrics = self.evaluate_performance(
                    {
                        "epochs": i,
                        "elbo": epoch["elbo"],
                        "div": epoch["div"],
                        "ll": epoch["ll"],
                    }
                )

                # Report performance.
                report = ""
                report += f"epochs: {metrics['epochs']} "
                report += f"elbo: {metrics['elbo']:.3f} "
                report += f"ll: {metrics['ll']:.3f} "
                report += f"div: {metrics['div']:.3f} \n"
                for k, v in metrics.items():
                    performance_metrics[k].append(v)
                    if "mll" in k or "acc" in k:
                        report += f"{k}: {v:.3f} "

                tqdm.write(report)
                
                break

        # Log the training curves for this update.
        if self.config["log_training"]:
            self.log["training_curves"].append(training_metrics)
        if self.config["log_performance"]:
            self.log["performance_curves"].append(performance_metrics)

        # Create non-trainable copy to send back to server.
        if self.config["early_stopping"].stash_model:
            q_new = self.config["early_stopping"].best_model
        else:
            q_new = q.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                q_new,
                q_old,
                damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
                update_log_coeff=self.config["update_log_coeff"],
            )

            return q_new, t_new

        else:
            return q_new, None

    def model_predict(self, x, **kwargs):
        """
        Returns the current models predictive posterior distribution.
        :return: ∫ p(y | θ, x) q(θ) dθ.
        """
        return self.model(x, self.q, **kwargs)


class ClientBayesianHypers(Client):
    """
    PVI client with Bayesian treatment of model hyperparameters.
    """
    """def __init__(self, teps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set likelihood approximating term
        self.teps = teps"""

    def __init__(self, data, model, teps=None, *args, **kwargs):
        super().__init__(data, model, *args, **kwargs)

        # Set likelihood approximating term
        self.teps = teps

    # New implementation of __init(...)
    """def __init__(self, data, model, t=None, teps=None, config=None, val_data=None):

        if config is None:
            config = {}

        #self._config = self.get_default_config()
        self._config = config

        # Set data partition and likelihood
        self.data = data
        self.model = model

        # Set likelihood approximating term
        self.t = t
        self.teps = teps

        self.log = defaultdict(list)
        self._can_update = True

        # Validation dataset.
        self.val_data = val_data"""

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "num_elbo_hyper_samples": 1,
        }

    def update_q(self, q, qeps, init_q=None, init_qeps=None):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """
        # Pass a trainable copy to optimise.
        q_new, qeps_new, self.t, self.teps = self.gradient_based_update(
            q.trainable_copy(), qeps.trainable_copy()
        )

        return q_new, qeps_new, self.t, self.teps

    def gradient_based_update(self, q, qeps):
        # Cannot update during optimisation.
        print("Entering Client Bayesian Hypers")
        self._can_update = False

        if self.t is None:
            # Old posterior = prior, make non-trainable.
            q_cav = q.non_trainable_copy()
            qeps_cav = qeps.non_trainable_copy()
        else:
            q_cav = q.non_trainable_copy()
            q_cav.nat_params = {
                k: v - self.t.nat_params[k] for k, v in q_cav.nat_params.items()
            }

            qeps_cav = qeps.non_trainable_copy()
            for k1, v1 in qeps.distributions.items():
                qeps.distributions[k1].nat_params = {
                    k2: v2 - self.teps.factors[k1].nat_params[k2]
                    for k2, v2 in v1.nat_params.items()
                }

        parameters = list(q.parameters()) + qeps.parameters()

        # Reset optimiser. Parameters are those of q(θ) and q(ε).
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"]
        )

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(
            tensor_dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        # Dict for logging optimisation progress
        training_curve = {
            "elbo": [],
            "div": [],
            "kleps": [],
            "ll": [],
            "logt": [],
            "logteps": [],
        }

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch")
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = {
                "elbo": 0,
                "div": 0,
                "kleps": 0,
                "ll": 0,
                "logt": 0,
                "logteps": 0,
            }

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute KL divergence between q and p.
                kl = q.kl_divergence(q_cav).sum() / len(x)
                kleps = sum(qeps.kl_divergence(qeps_cav).values()) / len(x)

                # Estimate E_q[log p(y | x, θ, ε)].
                ll = 0
                for _ in range(self.config["num_elbo_hyper_samples"]):
                    eps = qeps.rsample()
                    self.model.hyperparameters = eps
                    ll += self.model.expected_log_likelihood(
                        batch, q, self.config["num_elbo_samples"]
                    ).sum()

                ll /= self.config["num_elbo_hyper_samples"] * len(x_batch)

                if self.t is not None:
                    logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                    logteps = sum(
                        self.teps.eqlogt(qeps, self.config["num_elbo_samples"]).values()
                    )
                    logt /= len(x)
                    logteps /= len(x)

                    # loss = kl + kleps - ll + logt - logteps

                loss = kl + kleps - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["kleps"] += kleps.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

                if self.t is not None:
                    epoch["logt"] += logt.item() / len(loader)
                    epoch["logteps"] += logteps.item() / len(loader)

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["kleps"].append(epoch["kleps"])
            training_curve["ll"].append(epoch["ll"])

            if self.t is not None:
                training_curve["logt"].append(epoch["logt"])
                training_curve["logteps"].append(epoch["logteps"])

            if i % self.config["print_epochs"] == 0:
                logger.debug(
                    f"ELBO: {epoch['elbo']:.3f}, "
                    f"LL: {epoch['ll']:.3f}, "
                    f"KL: {epoch['kl']:.3f}, "
                    f"KL eps: {epoch['kleps']:.3f}, "
                    f"log t: {epoch['logt']: .3f},"
                    f"log teps: {epoch['logteps']: .3f},"
                    f"Epochs: {i}."
                )

            epoch_iter.set_postfix(
                elbo=epoch["elbo"],
                kl=epoch["kl"],
                ll=epoch["ll"],
                kleps=epoch["kleps"],
                logt=epoch["logt"],
                logteps=epoch["logteps"],
            )

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)

        # Create non_trainable_copy to send back to server.
        q_new = q.non_trainable_copy()
        qeps_new = qeps.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                q,
                q_cav,
                damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
            )
            teps_new = self.teps.compute_refined_factor(
                qeps,
                qeps_cav,
                damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
            )

            return q_new, qeps_new, t_new, teps_new

        else:
            return q_new, qeps_new, None, None

    def model_predict(self, x, **kwargs):
        raise NotImplementedError
