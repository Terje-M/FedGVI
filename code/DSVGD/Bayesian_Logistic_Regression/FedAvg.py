import torch
from torch.distributions.normal import Normal
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from torch.distributions.multivariate_normal import MultivariateNormal
from .Library.general_functions import sigmoid
from .Library.bayesian_logistic_regression import BayesianLR
"""
    Bayesian Logistic Regression using FedAvg (McMahan et al., 2017).
    Kassab and Simeone (2022) use a similar setting to Gershman et al., 2012.
    They schedule one agent/client at a time for fairness with DSVGD.
"""
class FedAvg():

    def __init__(self):
        self.alpha = 1.  # temperature parameter multiplying divergence of posterior with prior
        self.nb_exp = 10  # number of experiments to average upon
        self.avg_accuracy = 0  # average accuracy after iterating nb_exp times through dataset
        self.avg_llh = 0  # average log-likelihood after iterating nb_exp times through dataset
        self.nb_global = 10  # number of global iterations
        self.nb_iter = 500  # number of local iterations per client
        self.K = 2  # number of agents
        self.alpha_ada = 0.05  # constant rate used in AdaGrad
        self.epsilon_ada = 10 ** (-9)  # fudge factor for adagrad
        self.N = 1  # number of particles = 1 as it is a frequentist approach
        self.batchsize = 1024  # size of a batch
        self.betta = 0.9  # for momentum update
        self.array_accuracy = np.zeros(self.nb_global)
        self.array_llh = np.zeros(self.nb_global)

        self.a, self.b = 1., 1/0.01  # b = rate = 1/scale

    def fed_avg_client(self,alpha_ada, betta, epsilon_ada, theta, nb_iter, X, y, batchsize):

        # sum of gradients used in AdaGrad
        sum_squared_grad = 0
        for t in range(nb_iter):

            batch = [i % X.shape[0] for i in range(t * batchsize, (t + 1) * batchsize)]
            ridx = np.random.permutation(batch)

            Xs = X[ridx, :]
            ys = y[ridx]

            # compute gradient
            A = ys * np.matmul(theta, Xs.T)
            A = (- (1 - sigmoid(A)) * ys)

            delta_theta = X.shape[0] / Xs.shape[0] * (A.T * Xs).sum(axis=0)  # re-scaled gradient

            if t == 0:
                sum_squared_grad = delta_theta ** 2
            else:
                sum_squared_grad = betta * sum_squared_grad + (1 - betta) * (delta_theta ** 2)

            ada_step = alpha_ada / (epsilon_ada + np.sqrt(sum_squared_grad))

            theta = theta - ada_step * delta_theta
        return theta


    def server(self,alpha_ada, betta, epsilon_ada, a, b, theta, nb_iter, nb_global, K, y, X, y_test, X_test, batchsize):
        """
        Implements Federated Averaging (McMahan et al., 2017) with round robin scheduling
        of one worker/agent per global iteration
        """

        tot_size = X.shape[0]  # total size of datasets
        acc = np.zeros(nb_global)
        llh = np.zeros(nb_global)
        model = BayesianLR(X, y)
        for i in range(nb_global):

            # pick one agent in a round robin manner
            curr_agent = i % K

            # local training dataset
            X_curr, y_curr = X[curr_agent*X.shape[0]//K: ((curr_agent + 1)*X.shape[0]) // K, :], y[curr_agent*X.shape[0]//K : ((curr_agent + 1)*X.shape[0]) // K]
            n_curr = X_curr.shape[0]

            # update global parameters after re-weighting with respect to local dataset sizes of each agent
            theta = (n_curr/tot_size) * self.fed_avg_client(alpha_ada, betta, epsilon_ada, theta, nb_iter, X_curr, y_curr, batchsize) + (1-n_curr/tot_size) * theta

            acc[i], llh[i] = model.evaluation(theta, X_test, y_test)

        return acc, llh


    def run(self):
        np.random.seed(0)
        torch.random.manual_seed(0)

        seeds = [2, 13, 21, 42, 71, 97, 151, 227, 331, 397]

        ''' Read and preprocess data from covertype dataset'''
        data = scipy.io.loadmat('../data/covertype.mat')
        X_input = data['covtype'][:, 1:]
        y_input = data['covtype'][:, 0]
        y_input[y_input == 2] = -1  # please ensure labels are in {-1, +1}

        d = X_input.shape[1]  # dimension of each particle = dimension of a data point

        accuracies = []
        log_likes = []

        for exp in range(self.nb_exp):
            print('Trial ', exp + 1)

            # split the dataset into training and testing
            X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=seeds[exp])

            p_etta = torch.distributions.gamma.Gamma(torch.tensor([self.a]), torch.tensor([self.b]))
            etta = p_etta.rsample(torch.Size([self.N]))

            # initialize particles using mutlivariate normal
            particles = torch.zeros(torch.Size([self.N, d]))
            mean_0 = torch.zeros(d)
            for i in range(self.N):
                particles[i, :] = MultivariateNormal(mean_0, 1/etta[i]*torch.eye(d, d)).rsample()

            ''' Run Federated Averaging server with Round Robin (RR) scheduling '''
            curr_accuracy, curr_llh =\
                self.server(self.alpha_ada, self.betta, self.epsilon_ada, self.a, self.b, particles.detach().numpy(), self.nb_iter, self.nb_global, self.K, y_train, X_train, y_test, X_test, self.batchsize)

            self.array_accuracy += curr_accuracy
            self.array_llh += curr_llh

            accuracies.append(curr_accuracy)
            log_likes.append(curr_llh)

        print('BLR accuracy with FedAvg as function of comm. rounds = ', repr(self.array_accuracy / self.nb_exp))
        print('BLR average llh with FedAvg as function of comm. rounds = ', repr(self.array_llh / self.nb_exp))

        return {
            "validation": accuracies,
            "log_like": log_likes,
        }