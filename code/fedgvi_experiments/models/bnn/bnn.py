import torch
from torch import nn

import numpy as np

from fedgvi_experiments.models.base import Model

from abc import ABC, abstractmethod


class FullyConnectedBNN(Model, nn.Module, ABC):

    conjugate_family = None

    def __init__(self, activation=nn.ReLU(), **kwargs):

        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self.activation = activation

    def get_default_nat_params(self):

        return {
            "np1": torch.zeros(size=(self.num_parameters,)),
            "np2": -0.5 * 1e6 * torch.ones(size=(self.num_parameters,)),
        }

    def get_default_config(self):
        return {
            "num_predictive_samples": 10,
            "latent_dim": 128,
            "num_layers": 2,
        }

    def get_default_hyperparameters(self):
        """
        :return: A default set of ε for the model.
        """
        return {}

    def forward(self, x, q, **kwargs):

        # Number of θ samples to draw
        num_pred_samples = self.config["num_predictive_samples"]
        theta = q.distribution.sample((num_pred_samples,))

        # Collection of output distributions, one for each θ, x pair
        # Distribution assumed to be of shape (S, N, D).
        qy = self.likelihood_forward(x, theta, samples_first=False)

        # Predictive is a mixture of predictive distributions with equal
        # weights.
        mix = torch.distributions.Categorical(
            logits=torch.ones(size=qy.batch_shape).to(x)
        )
        qy = torch.distributions.MixtureSameFamily(mix, qy)

        return qy

    def likelihood_forward(self, x, theta, samples_first=True):

        assert len(x.shape) in [1, 2], "x must be (N, D)."
        assert len(theta.shape) in [1, 2], "theta must be (S, K)."

        if len(x.shape) == 1:
            x = x[None, :]

        if len(theta.shape) == 1:
            theta = theta[None, :]

        # Converts θ-vectors to tensors, shaped as expected by the network.
        # i.e. (S, D_in + 1, D_out).
        theta = self.reshape_theta(theta)
        for i, W in enumerate(theta):
            # Bias term.
            x = torch.cat([x, torch.ones((*x.shape[:-1], 1)).to(x)], dim=-1)
            x = x.matmul(W)

            # Don't apply ReLU to final layer.
            if i < len(theta) - 1:
                x = self.activation(x)

        return self.pred_dist_from_tensor(x, samples_first=samples_first)

    def reshape_theta(self, theta):

        # Number of Monte Carlo samples of parameters.
        num_samples = theta.shape[0]

        # Check total number of parameters in network equals size of theta.
        assert self.num_parameters == theta.shape[-1]

        # Indices to slice the theta tensor at
        slices = np.cumsum([0] + self.sizes)

        theta = [
            torch.reshape(theta[:, s1:s2], [num_samples] + list(shape))
            for shape, s1, s2 in zip(self.shapes, slices[:-1], slices[1:])
        ]

        return theta

    def conjugate_update(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def shapes(self):
        shapes = []

        for i in range(self.config["num_layers"] + 1):
            if i == 0:
                shapes.append((self.config["input_dim"] + 1, self.config["latent_dim"]))
            elif i == self.config["num_layers"]:
                # Weight matrix.
                shapes.append(
                    (self.config["latent_dim"] + 1, self.config["output_dim"])
                )
            else:
                # Weight matrix.
                shapes.append(
                    (self.config["latent_dim"] + 1, self.config["latent_dim"])
                )

        return shapes

    @abstractmethod
    def pred_dist_from_tensor(self, tensor, samples_first=False):
        raise NotImplementedError

    @property
    def sizes(self):
        return [np.prod(shape) for shape in self.shapes]

    @property
    def num_parameters(self):
        return sum(self.sizes)


class RegressionBNN(FullyConnectedBNN):
    def pred_dist_from_tensor(self, tensor, samples_first=True):

        loc = tensor[:, :, : self.output_dim]
        scale = torch.exp(tensor[:, :, self.output_dim :])

        return torch.distributions.normal.Normal(loc=loc, scale=scale)


class ClassificationBNN(FullyConnectedBNN):
    def pred_dist_from_tensor(self, tensor, samples_first=True):

        if not samples_first:
            tensor = torch.transpose(tensor, 0, 1)

        return torch.distributions.Categorical(logits=tensor)


class FullyConnectedBNNLocalRepam(FullyConnectedBNN):
    def local_repam_forward(self, x, q, num_samples=None):
        """
        Returns samples from the predictive posterior distribution of a
        Bayesian neural network, sampling activations instead of weights.
        :param x: The input locations to make predictions at.
        :param q: The approximate posterior distribution q(θ).
        :param num_samples: The number of samples to estimate the predictive
        posterior distribution with.
        :return: Samples from ∫ p(y | θ, x) q(θ) dθ.
        """
        if num_samples is None:
            num_samples = self.config["num_predictive_samples"]

        # Compute activation distribution at each layer and sample.
        num_params = np.cumsum([0] + self.sizes)
        for i, shape in enumerate(self.shapes):
            # Bias term.
            x = torch.cat([x, torch.ones((*x.shape[:-1], 1)).to(x)], dim=-1)

            qw_loc = q.std_params["loc"][
                ..., num_params[i] : num_params[i + 1]
            ].reshape(q.std_params["loc"].shape[:-1] + shape)
            qw_scale = q.std_params["scale"][
                ..., num_params[i] : num_params[i + 1]
            ].reshape(q.std_params["loc"].shape[:-1] + shape)

            # Layer's q(h). Add jitter to scale to prevent numerical errors
            # during backward pass.
            # (batch_size, dim_out).
            #print(x.dtype, qw_loc.dtype)
            qh_loc = x.matmul(qw_loc)
            qh_scale = ((x ** 2).matmul(qw_scale ** 2) + 1e-6) ** 0.5

            # Use different random sample for each datapoint to reduce
            # covariance.
            if i == 0:
                # (num_samples, batch_size, dim_out).
                qh_eps = qh_loc.new(num_samples, *qh_loc.shape).normal_()
                x = qh_loc + qh_scale * qh_eps
            else:
                # (num_samples, batch_size, dim_out).
                qh_eps = qh_loc.new(*qh_loc.shape).normal_()
                x = qh_loc + qh_scale * qh_eps

            if i < self.config["num_layers"]:
                x = self.activation(x)

        return x

    # def forward(self, x, q, num_samples=None, **kwargs):
    #     """
    #     Returns the predictive posterior distribution of a Bayesian neural
    #     network using by sampling activations instead of weights.
    #     :param x: The input locations to make predictions at.
    #     :param q: The approximate posterior distribution q(θ).
    #     :param num_samples: The number oof samples to estimate the predictive
    #     posterior distribution with.
    #     :return: ∫ p(y | θ, x) q(θ) dθ.
    #     """
    #     # Get outputs, sampled using local reparameterisation.
    #     qy = self.local_repam_forward(x, q, samples_first=False,
    #                                   num_samples=num_samples)
    #
    #     # Create equal mixture of distributions.
    #     mix = torch.distributions.Categorical(
    #         logits=torch.ones(size=qy.batch_shape).to(x.device))
    #     qy = torch.distributions.MixtureSameFamily(mix, qy)
    #
    #     return qy

    def expected_log_likelihood(self, data, q, num_samples=1):
        """
        Computes the expected log likelihood of the data under q(θ) using the
        local reparameterisation trick.
        :param data: The data to compute the conjugate update with.
        :param q: The current global posterior q(θ).
        :param num_samples: The number of samples to estimate the expected
        log-likelihood with.
        :return: ∫ q(θ) log p(y | x, θ) dθ.
        """
        x = data["x"]
        y = data["y"]

        h = self.local_repam_forward(x, q, num_samples=num_samples)
        qy = self.pred_dist_from_tensor(h, samples_first=True)

        return qy.log_prob(y).mean(0)


class ClassificationBNNLocalRepam(FullyConnectedBNNLocalRepam):
    def pred_dist_from_tensor(self, tensor, samples_first=True):

        if not samples_first:
            tensor = torch.transpose(tensor, 0, 1)

        if tensor.shape[-1] == 1:
            return torch.distributions.Bernoulli(logits=tensor)
        else:
            return torch.distributions.Categorical(logits=tensor)
    
    def expected_beta_loss(self, data, q, num_samples=1, beta=1.1):
        # Compute the beta loss function using the parametrisation of Knoblauch et al. (2022)
        # We find E_q(θ)[Σ_i^n L_{β,p}(y_i|θ,x_i)]
        # where we sample from q(θ) and then evaluate for β > 1
        #
        # - 1/(β-1) * p(y_i | x_i, θ)^(β-1) + 1/β * Σ_{j=1}^C p(ŷ_j | x_i, θ)^β
        #
        # which recovers the negative log likelihood as β → 1
        
        assert beta >= 1, f"Trying to do beta loss with beta value {beta} smaller than 1"
        
        if beta == 1:
            return self.expected_log_likelihood(self, data, q, num_samples=1)
        #print("Computing beta loss")
        x = data["x"]
        y = data["y"]

        # sample weights for each num_samples and for each data point x
        h = self.local_repam_forward(x, q, num_samples=num_samples)

        #sample outputs associated with weights
        qy = self.pred_dist_from_tensor(h, samples_first=True)

        term1 = torch.exp(((beta) * qy.log_prob(y))).mean(0) * beta / (beta - 1)

        term2 = 0.0
        for i in range(h.shape[-1]):
            ty = i * torch.ones_like(y)
            term2 += torch.exp(((beta - 1) * qy.log_prob(ty))).mean(0) / beta

        ret = - term1 + term2
        return ret

    def robust_cross_entropy(self, data, q, num_samples=1, beta=0.9):
        x = data["x"]
        y = data["y"]

        # sample weights for each num_samples and for each data point x
        h = self.local_repam_forward(x, q, num_samples=num_samples)

        #sample outputs associated with weights
        qy = self.pred_dist_from_tensor(h, samples_first=True)

        return (1 - torch.exp(beta * qy.log_prob(y))) / beta



class RegressionBNNLocalRepam(FullyConnectedBNNLocalRepam):
    def pred_dist_from_tensor(self, tensor, samples_first=True):

        loc = tensor[..., : self.config["output_dim"] // 2]
        scale = torch.exp(tensor[..., self.config["output_dim"] // 2 :])

        return torch.distributions.normal.Normal(loc=loc, scale=scale)
