import torch
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split

from fedgvi_experiments.logistic_regression.LogRegServer import LogRegServer
from fedgvi_experiments.utils.helper_functions import helper_functions, Gaussian

datasets = ["covertype", "synthetic_1", "synthetic_2", "synthetic_3", "synthetic_4"]
torch.set_default_dtype(torch.float64)
# =============================================================================
# Logistic Regression Experiments 
# This code base was created by the authors (Mildner et al., 2025)
# and follows the set up of Kassab and Simeone (2022)
# =============================================================================
#
# To reproduce the experiments in the paper simply run 'predefined_experiments'
# The correpsonding Jupyter Notebook file aggregates the results and produces the figures

class LogReg:
    
    # To Run 
    def predefined_experiments(self):
        run1 = self.run(num_clients=10,
                        data_set="covertype")
        
        run2 = self.run(num_clients=50,
                        data_set="covertype")
        
        run3 = self.run(num_clients=10,
                        data_set="covertype",
                        heterogeneity=True)
        
        run4 = self.run(num_clients=50,
                        data_set="covertype",
                        heterogeneity=True)
        
        return [run1, run2, run3, run4]
    
    def run(
            self,
            num_clients,
            data_set="covertype",
            augment=False,
            train_test_split = 0.2,
            contamination={"contaminate": False, "type": None, "epsilon": 0.2},
            heterogeneity=False,
            server_iterations=10,
            optim_epochs=200,
            lr = 5e-3,
            monte_carlo_samples=100,
            minibatch_size = np.inf,
            client_batch_frac = 1.0,
            diff_prior_loc = None,
            diff_prior_cov = None,
            client_div = "KLD",
            client_div_param = None,
            client_loss = "nll",
            client_score_fct = None,
            client_loss_param = None,
            seed=42,
            return_data=False,
            global_div="KLD",
            global_div_param=None,
        ):
        
        training_set, validation_set = self.data_init(data_set, contamination, augment, train_test_split, seed)

        dim = self.get_data_dimension(training_set)
        N = len(training_set["y"])

        clients = self.client_init(num_clients, training_set, dim, heterogeneity, client_div, client_div_param, \
                                   client_loss, client_loss_param, client_score_fct, seed)

        server_parameters = self.parameter_init(dim, N, server_iterations, optim_epochs, monte_carlo_samples, minibatch_size, lr, seed, global_div, global_div_param)

        prior_dist = self.prior_dist(dim, loc_=diff_prior_loc, cov_=diff_prior_cov)

        mb = False if minibatch_size == np.inf else True
        ret_dict = LogRegServer().FedGVI(prior_dist, clients, server_parameters, validation_set, minibatch=mb, batched_clients=client_batch_frac)

        if return_data:
            ret_dict["train_data"] = training_set
            ret_dict["test_data"] = validation_set

        return ret_dict
    
    def client_init(self, num_clients, training_set, dim, heterogeneity, div, div_param, loss_fct, loss_param, score_fct, seed):
        
        client_data = self.split_client_train_sets(training_set, num_clients, heterogeneity, seed)

        clients = []
        for i in range(num_clients):
            clients.append({
                "client_idx": i,
                "data": client_data[i],
                "mean": torch.zeros(dim),
                "variance": np.inf * torch.ones(dim),
                "variance_inverse": torch.zeros(dim),
                "iteration": 0,
                "Divergence": div,
                "div_param": div_param,
                "loss": loss_fct,
                "loss_param": loss_param,
                "score_function": score_fct,
            })

        return clients
    
    def parameter_init(self, dim, N, server_iterations, optim_epochs, monte_carlo_samples, minibatch_size, lr, seed, global_div, global_div_param):
        return {
            "D": dim,
            "N": N,
            "Epochs": server_iterations,
            "global_div": global_div,
            "global_div_param": global_div_param,
            "lr": lr,
            "num_samples": monte_carlo_samples,
            "optim_epochs": optim_epochs,
            "batch_size": minibatch_size,
            "default_seed": seed,
        }
    
    # covariance matrices are parametrised as vectors and taken to be diagonal
    def prior_dist(self, d, loc_=None, cov_=None):
        if loc_ != None:
            loc = loc_
        else:   
            loc = torch.zeros(d)

        if cov_ != None:
            if cov_.shape[0] == d:
                cov = cov_
            else:
                cov = cov_ * torch.ones(d)
        else:
            cov = torch.ones(d)

        assert loc.shape == cov.shape, "Prior mean and covariance matrices are not of the same dimension"

        return {
            "loc": loc,
            "var": cov
        }
    
    def get_data_dimension(self, training_set):
        d = training_set["x"][0].shape[0]
        return d
    
    def data_init(self, data_set, contamination, augment=False, test_split=0.2, seed=42):
        assert data_set in datasets, f"Data set name '{data_set}' not recognised. Please choose one of the following: {datasets}."
        if data_set == "covertype":
            data = scipy.io.loadmat('./../data/covertype.mat')
            X_input_ = data['covtype'][:, 1:]
            if augment:
                X_input = np.column_stack((np.ones(len(X_input_)), X_input_))# If w^T X + b is desired
            else:
                X_input = X_input_
            y_input = data['covtype'][:, 0]
            y_input[y_input == 2] = 0  # ensure labels are in {0, 1} otherwise logistic loss breaks but we relabel as needed
            
            X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=test_split, random_state=seed)

            training_data = {
                "x": torch.tensor(X_train),
                "y": torch.tensor(y_train)
            }
            validation_data = {
                "x": torch.tensor(X_test),
                "y": torch.tensor(y_test)
            }
        elif data_set == "synthetic_1":
            
            np.random.seed(83_507_569) #Fix the data set, but not the train-test split. We've generated a random seed for this.

            trainx_ = np.vstack((np.random.randn(100,2),np.random.randn(100,2)+3))
            trainy = np.hstack((-1*np.ones(100,),np.ones(100,)))[:,None]

            if augment:
                trainx = np.column_stack((np.ones(len(trainx_)), trainx_))# If w^T X + b is desired
            else:
                trainx = trainx_

            tx = torch.tensor(trainx)
            ty = torch.tensor(trainy)

            ty = ty.squeeze()
            ty[ty == -1] = 0 # {0,1} data required

            #Rescaling the dataset, but not to zero mean:
            mu = tx.mean(0)

            for i in range(len(ty)):
                temp_x = tx[i][0] - mu[0] + 1
                temp_y = tx[i][1] - mu[1] - 1
                tx[i][0] = temp_x
                tx[i][1] = temp_y

            X_train, X_test, y_train, y_test = train_test_split(tx, ty, test_size=test_split, random_state=seed)

            training_data = {
                "x": torch.tensor(X_train),
                "y": torch.tensor(y_train)
            }
            validation_data = {
                "x": torch.tensor(X_test),
                "y": torch.tensor(y_test)
            } 
        elif data_set == "synthetic_2":
            
            np.random.seed(83_507_569) #Fix the data set, but not the train-test split. We've generated a random seed for this.

            trainx_ = np.vstack((np.random.randn(100,2)-1,np.random.randn(100,2)+4))
            trainy = np.hstack((-1*np.ones(100,),np.ones(100,)))[:,None]

            trainx = np.column_stack((np.ones(len(trainx_)), trainx_))# If w^T X + b is desired
            
            tx = torch.tensor(trainx)
            ty = torch.tensor(trainy)

            ty = ty.squeeze()
            ty[ty == -1] = 0 # {0,1} data required
            
            X_train, X_test, y_train, y_test = train_test_split(tx, ty, test_size=test_split, random_state=seed)

            training_data = {
                "x": torch.tensor(X_train),
                "y": torch.tensor(y_train)
            }
            validation_data = {
                "x": torch.tensor(X_test),
                "y": torch.tensor(y_test)
            }
        elif data_set == "synthetic_3":
            np.random.seed(83_507_569)

            mean_1 = torch.tensor([2.,4.])
            mean_2 = torch.tensor([0.,0.])
            cov_1 = torch.tensor([[2.,0.5],[0.5,1.]])
            cov_2 = torch.tensor([[0.8,-0.4],[-0.4,1.]])

            torch.manual_seed(78)
            samples = Gaussian.HeterogenousSampleMix([mean_1,mean_2], [cov_1,cov_2], torch.tensor([.5,.5]), 60)

            labels = np.hstack((np.ones(len(samples[0]),),-1*np.ones(len(samples[1]),)))[:,None]
            trainx_ = torch.cat((samples[0], samples[1]))

            if augment:
                trainx = np.column_stack((np.ones(len(trainx_)), trainx_))# If w^T X + b is desired
            else:
                trainx = trainx_
            
            tx = torch.tensor(trainx)
            ty = torch.tensor(labels).squeeze()
            ty[ty == -1] = 0
            
            
            if contamination["contaminate"] == True and contamination["type"] == "adversarial1":
                num_eps = int(np.ceil(contamination["epsilon"] * len(ty) / (1 - contamination["epsilon"])))
                adv_x_ = Gaussian.GaussianSamples(torch.tensor([-1.0,5.]), 0.5*torch.eye(2), num_eps)
                adv_y = torch.zeros(num_eps)

                if augment:
                    adv_x = torch.tensor(np.column_stack((np.ones(len(adv_x_)), adv_x_)))
                else:
                    adv_x = torch.tensor(adv_x_)

                tx_train = torch.cat((tx, adv_x))
                ty_train = torch.cat((ty, adv_y))
            elif contamination["contaminate"] == True and contamination["type"] == "adversarial2":
                num_eps = int(np.ceil(contamination["epsilon"] * len(ty) / (1 - contamination["epsilon"])))
                adv_x_ = Gaussian.GaussianSamples(torch.tensor([4.0,4.]), 0.5*torch.eye(2), num_eps)
                adv_y = torch.zeros(num_eps)

                if augment:
                    adv_x = torch.tensor(np.column_stack((np.ones(len(adv_x_)), adv_x_)))
                else:
                    adv_x = torch.tensor(adv_x_)

                tx_train = torch.cat((tx, adv_x))
                ty_train = torch.cat((ty, adv_y))
            else:
                tx_train = tx
                ty_train = ty

            training_data = {
                "x": torch.tensor(tx_train),
                "y": torch.tensor(ty_train)
            }
            validation_data = {
                "x": torch.tensor(tx),
                "y": torch.tensor(ty)
            }
        elif data_set == "synthetic_4":
            np.random.seed(83_507_569)

            mean_1 = torch.tensor([3.,4.])
            mean_2 = torch.tensor([0.,0.])
            cov_1 = torch.tensor([[1.,0.35],[0.35,.75]])
            cov_2 = torch.tensor([[0.8,-0.4],[-0.4,1.]])

            torch.manual_seed(169)
            # This results in a linearly separable data set.
            samples = Gaussian.HeterogenousSampleMix([mean_1,mean_2], [cov_1,cov_2], torch.tensor([.5,.5]), 100)

            labels = np.hstack((np.ones(len(samples[0]),),-1*np.ones(len(samples[1]),)))[:,None]
            trainx_ = torch.cat((samples[0], samples[1]))

            trainx = np.column_stack((np.ones(len(trainx_)), trainx_))# If w^T X + b is desired
            
            tx = torch.tensor(trainx)
            ty = torch.tensor(labels).squeeze()
            ty[ty == -1] = 0
            
            if contamination["contaminate"] == True and contamination["type"] == "adversarial1":
                # We define the contamination data set of 6 contaminating points so that 5% of the entire data set are contaminated.
                torch.manual_seed(117)
                num_eps = int(np.ceil(0.05 * len(ty) / (1 - 0.05)))
                adv_x_ = Gaussian.GaussianSamples(torch.tensor([1.0,5.0]), torch.tensor([[0.15,0.1],[0.1,0.15]]), num_eps)
                adv_y = torch.zeros(num_eps)

                adv_x = torch.tensor(np.column_stack((np.ones(len(adv_x_)), adv_x_)))
                
                tx_train = torch.cat((tx, adv_x))
                ty_train = torch.cat((ty, adv_y))
            else:
                tx_train = tx
                ty_train = ty

            training_data = {
                "x": torch.tensor(tx_train),
                "y": torch.tensor(ty_train)
            }
            validation_data = {
                "x": torch.tensor(tx),
                "y": torch.tensor(ty)
            }
        
        if contamination["contaminate"] == True:
            training_data = self.introduce_label_contamination(contamination, training_data)
    
        return training_data, validation_data
    
    def split_client_train_sets(self, data_set, num_clients, heterogeneity, seed):
        if heterogeneity == False:
            np.random.seed(seed)
    
            perm = np.random.permutation(len(data_set["x"]))
            client_data = []
            for i in range(num_clients):
                client_idxs = perm[i::num_clients]
                client_data.append({"x": data_set["x"][client_idxs], "y": data_set["y"][client_idxs]})
        else:
            raise NotImplementedError
        
        return client_data
    
    def introduce_label_contamination(self, params, data_set, seed=42):
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        if params["type"] == "random":
            total = len(data_set["y"])
            eps_frac = int(np.round(params["epsilon"] * total))
            
            corrupted_indices = rng.choice(total, eps_frac, replace=False)

            # Assuming labels are {0,1}
            print(data_set["y"][corrupted_indices])
            data_set["y"][corrupted_indices] = 1 - data_set["y"][corrupted_indices]
            print(data_set["y"][corrupted_indices])
        elif params["type"] == "one_sided":
            zeros = [key for key, val in enumerate(data_set["y"])
                                    if val == 0]
            ones = [key for key, val in enumerate(data_set["y"])
                                    if val == 1]
            
            if len(zeros) >= len(ones):
                data = torch.tensor(zeros)
            else:
                data = torch.tensor(ones)

            total1 = len(data_set["y"])
            total2 = len(data)
            
            eps_frac = int(np.round(params["epsilon"] * total1))
            if eps_frac > total2:
                eps_frac = total2

            corrupted_indices = rng.choice(total2, eps_frac, replace=False)
            
            print(data_set["y"][data[corrupted_indices]])
            data_set["y"][data[corrupted_indices]] = 1 - data_set["y"][data[corrupted_indices]]
            print(data_set["y"][data[corrupted_indices]])
        elif params["type"] == "concentrated":
            zeros = [key for key, val in enumerate(data_set["y"])
                                    if val == 0]
            ones = [key for key, val in enumerate(data_set["y"])
                                    if val == 1]
            
        elif params["type"] == "adversarial1" or params["type"] == "adversarial2":
            # Contamination already occured
            return data_set
        else:
            raise NotImplementedError
    
        return data_set
