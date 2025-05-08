from __future__ import division

import torch
import numpy as np
import matplotlib as plt
import math
from sklearn.model_selection import train_test_split 

class helper_functions:
    
    def homogeneous_split(data, partitions):
        rng = np.random.default_rng()
        perm = rng.permutation(len(data))
        
        client_data = []
        for i in range(partitions):
            client_idx = perm[i::partitions]
            client_data.append(data[client_idx])
        
        return client_data

    def get_batch(data, length):
        assert length > 0, "Can't use batch of size 0"
        rng = np.random.default_rng()
        perm = rng.permutation(len(data))
        idx_0 = perm[0]
        batch = torch.atleast_2d(data[idx_0])
        l = min(len(data), length)
        
        for i in range(1, l):
            client_idx = perm[i]
            samp = torch.atleast_2d(data[client_idx])
            batch = torch.cat((batch, samp), 0)
        return batch
    
    def minibatch_of_indices(fraction, num=None, set=None, seed=42):
        assert num == None or set == None, "Can't have both a set to choose from and a number of indices." 
        assert num != None or set != None, "Need a target to choose from."

        rng = np.random.default_rng(seed=seed)

        if set == None:
            size = int(math.floor(fraction * num))
            arr = rng.choice(num, size=size, replace=False)
        else:
            size = int(math.floor(fraction * len(set)))
            arr = rng.choice(set, size=size, replace=False)

        arr.sort()

        return arr
    
    def get_xy_batch(data, batch_length, seed=42):
        assert batch_length >= 1

        if batch_length >= len(data["y"]): 
            return data

        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(len(data["y"]), size=int(batch_length), replace=False)

        x_batch = data["x"][indices]
        y_batch = data["y"][indices]
        
        return {
            "x": x_batch,
            "y": y_batch
        }

    # Will generate random sized partitions of the data, dependent on the size of the previous partitions
    def heterogeneous_split(data, partitions, seed=42):
        
        l = len(data)
        weights = []
        for i in range(partitions-1):
            torch.manual_seed(seed+i)
            w_i = torch.distributions.uniform.Uniform(torch.tensor([1.]), (l-partitions+i)).sample()
            w = torch.floor(w_i)
            weights.append(w.numpy())
            l -= w
        weights.append(l.numpy())
        
        client_data = []
        counter = 0
        for w in weights:
            client_x_n = data[counter:int(counter+w[0])]
            client_data.append(client_x_n)
            counter += w[0]
            counter = int(counter)
            
        return client_data
    
    def horizontally_partitioned_data(data, partitions):
        num = len(data)
        n = int(np.ceil(num / partitions))
        if len(data.shape) <2:
            data, _ = torch.sort(data)
            temp = torch.split(data, n)
        else:
            temp = torch.split(data, n)
        ret = []
        for t in temp:
            ret.append(t)
        return ret
       
    
    def plot_training(training_array):
        x_vals = np.arange(1, len(training_array)+1)
        #plt.figure(figsize=(6,4))
        plt.grid(visible=True)
        plt.plot(x_vals, training_array)
        plt.ylabel('ELBO Loss')
        plt.xlabel('Step')
        plt.show()


class BNN_helper_functions:
    def homogeneous_split(data, num_clients=100, seed=42):
        # Set numpy's random seed.
            np.random.seed(seed)
            
            perm = np.random.permutation(len(data["x"]))
            client_data = []
            for i in range(num_clients):
                client_idx = perm[i::num_clients]
                client_data.append({"x": data["x"][client_idx], "y": data["y"][client_idx]})
        
            return client_data
    
    def heterogeneous_split_80_20(data, num_classes, num_clients, seed=42):
        
        listed = np.argsort(data["y"])

        sorted_x = data["x"][listed[::1]]
        sorted_y = data["y"][listed[::1]]
        
        if num_clients != num_classes:
            print("Less clients than data classes, so cannot split each client to have 80/20 labels of two classes.")
            l = len(data["y"])
            client_size = int(np.ceil(l/num_clients))
            
            client_data = []
            
            for i in range(num_clients):        
                client_data.append({"x": sorted_x[i*client_size:(i+1)*client_size], 
                                    "y": sorted_y[i*client_size:(i+1)*client_size]})
            
            return client_data
        else:
            eighties = []
            twenties = []

            for i in range(num_classes):
                x80, y80, x20, y20 = train_test_split(sorted_x[sorted_y == i], sorted_y[sorted_y == i], random_state=seed, test_size=0.2)
                eighties.append({"x": x80, "y": y80})
                twenties.append({"x": x20, "y": y20})
            client_data = []
            
            for i in range(num_classes):
                client_data_i_x = torch.cat((eighties[i]["x"], twenties[(i+1) % num_classes]["x"]))
                client_data_i_y = torch.cat((eighties[i]["y"], twenties[(i+1) % num_classes]["y"]))
                np.random.seed(seed)
                perm = np.random.permutation(len(client_data_i_y))
                client_data.append({"x": client_data_i_x[perm], "y": client_data_i_y[perm]})

            np.random.seed(seed)
            perm_c = np.random.permutation(num_clients)
            return client_data[perm_c]

        return None

    def heterogeneous_split_random(data, partitions, seed=42):
        l = len(data["y"])
        np.random.seed(seed)
        idxs = np.random.choice(l, partitions -1)
        idxs.sort()

        client_data = []
        prev = 0
        for i in range(partitions):
            client = {"x": data["x"][prev:idxs[i]], "y": data["y"][prev:idxs[i]]}
            client_data.append(client)
            prev = idxs[i]
            
        return client_data

    def contaminate_labels(data, epsilon, seed=42, random=True):
        if epsilon == 0.0:
            return data
        
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        total = len(data["y"])
        eps_frac = int(np.round(epsilon * total))
        
        corrupted_indices = rng.choice(total, eps_frac, replace=False)

        if random:
            BNN_helper_functions.random_contamination(data, corrupted_indices, seed=seed)
        else:
            BNN_helper_functions.deterministic_contamination(data, corrupted_indices, seed=seed)
        
        return corrupted_indices

    def random_contamination(data, indices, seed=42):
        labels = list(range(int(min(data["y"])), int(max(data["y"]))+1))

        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        
        for idx in indices:
            other_labels = labels.copy()
            other_labels.remove(int(data["y"][idx]))
            y_new = rng.choice(other_labels, 1)
            data["y"][idx] = y_new[0]
        
        return data

    def deterministic_contamination(data, indices, seed=42):
        #Naive contamination
        labels = list(range(int(min(data["y"])), int(max(data["y"]))+1))
        
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        labels_new = {}
        for y in labels:
            t= labels.copy()
            t.remove(y)
            y_new = rng.choice(t, 1)
            labels_new.update({y: y_new[0]})
        
        for idx in indices:
            prev = int( data["y"][idx])
            data["y"][idx] = labels_new[prev]
            
        return data

class Gaussian:
    #Change to log sum exp instead of explicitly evaluating this
    
    def Gaussian(mean, cov, theta):
        assert cov.shape[0] > 0, "Dimensionality specified incorrectly"
        if cov.shape[0] == 1:
            return torch.exp(-(((theta - mean) ** 2)/(2 * cov))) / ((torch.abs(2 * cov * np.pi)) ** 0.5)
        else:
            assert mean.shape[0] == cov.shape[1], "Not the same dimension, multiplication not possible"

            vector = theta-mean
            vector.unsqueeze(-1)

            # (\theta - mean)^T \Sigma ^ {-1} (\theta - mean)        
            part = torch.matmul(vector.unsqueeze(0), torch.linalg.solve(cov, vector.unsqueeze(-1)))

            det = torch.linalg.det(cov) ** 0.5

            coef_ = det * ((np.pi * 2) ** (cov.shape[0] / 2))
            
            coef = coef_ ** -1

            return (coef * torch.exp(- part / 2)).squeeze()
        
    def SphericalGaussian(mean, cov, theta, D):
        assert D > 0, "Negative dimension in spherical Gaussian"
        if D == 1:
            return torch.exp(-(((theta - mean) ** 2)/(2 * cov))) / ((torch.abs(2 * cov * np.pi)) ** 0.5)
        else:
            vector = theta-mean
            vector.unsqueeze(-1)

            # (\theta - mean)^T \Sigma ^ {-1} (\theta - mean)        
            part = torch.matmul(vector.unsqueeze(0), vector.unsqueeze(-1)) / cov

            coef_ = torch.pow(torch.abs(cov * np.pi * 2), (D / 2))

            coef = coef_ ** -1

            return (coef * torch.exp(- part / 2)).squeeze()
    
    def Normal(mean, cov, theta, D):
        if D is None:      
            return Gaussian.Gaussian(mean, cov, theta)
        else:
            return Gaussian.SphericalGaussian(mean, cov, theta, D)
    
    def GaussianMixture(mean_list, cov_list, theta, parameters, D=None):
        ret = 0.
        assert len(mean_list) == len(parameters), "Different list lengths"
        assert len(mean_list) == len(cov_list), "Different list lengths"
        
        param = 0.
        for w in parameters:
            param += w
            
        if param != 1:
            for k in range(len(parameters)):
                w = parameters[k]
                parameters[k] = w / param
                
        if D is None:      
            for i in range(len(parameters)):
                ret += parameters[i] * Gaussian.Gaussian(mean_list[i], cov_list[i], theta)
        else:
            for i in range(len(parameters)):
                ret += parameters[i] * Gaussian.SphericalGaussian(mean_list[i], cov_list[i], theta, D)
            
        return ret
    
    def SampleFromMixture(mean_list, cov_list, parameters, num_samples, include_mean=False):
        
        param = 0.
        for w in parameters:
            param += w
            
        if param != 1:
            for k in range(len(parameters)):
                w = parameters[k]
                parameters[k] = w / param
        
        sample_nums = torch.distributions.multinomial.Multinomial(num_samples, parameters).sample()
        component_sample_num = sample_nums.numpy()
        samples = torch.tensor([])
        mean = torch.zeros_like(mean_list[0])
        
        for i in range(len(parameters)):
            if cov_list[i].shape[0] > 1:    
                m_i = torch.distributions.multivariate_normal.MultivariateNormal(mean_list[i],
                                                                                 covariance_matrix=cov_list[i])
            else:
                m_i = torch.distributions.normal.Normal(mean_list[i],cov_list[i])
            
            if include_mean:
                mean = m_i.mean(dim=0)
                
            sample = m_i.sample((int(component_sample_num[i]),))
            samples = torch.cat((samples, sample), 0)
        if include_mean:     
            return samples, mean
        else:
            return samples
    
    def HeterogenousSampleMix(mean_list, cov_list, parameters, num_samples):
        
        param = 0.
        for w in parameters:
            param += w
            
        if param != 1:
            for k in range(len(parameters)):
                w = parameters[k]
                parameters[k] = w / param
        
        sample_nums = torch.distributions.multinomial.Multinomial(num_samples, 
                                                                           parameters).sample()
        component_sample_num = sample_nums.numpy()
        
        client_data =[]
        for i in range(len(parameters)):
            if cov_list[i].shape[0] > 1:    
                m_i = torch.distributions.multivariate_normal.MultivariateNormal(mean_list[i],
                                                                                 covariance_matrix=cov_list[i])
            else:
                m_i = torch.distributions.normal.Normal(mean_list[i],cov_list[i])
            
            sample = m_i.sample((int(component_sample_num[i]),))
            client_data.append(sample)
                
        return client_data
    
    def GaussianSamples(mean, covariance, num_samples):
        Normal = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=covariance)
        samples = Normal.sample((num_samples,))
        return samples