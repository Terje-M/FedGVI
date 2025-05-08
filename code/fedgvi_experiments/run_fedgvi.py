import sys
import os
import argparse
import  copy

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from pathlib import Path

data_dir = Path("../data")

import torch
import torch.utils.data
from torchvision import transforms, datasets
import numpy as np

from fedgvi_experiments.models import ClassificationBNNLocalRepam
from fedgvi_experiments.clients import Client
from fedgvi_experiments.servers import FedGVIServer
from fedgvi_experiments.distributions import MeanFieldGaussianDistribution, MeanFieldGaussianFactor
from fedgvi_experiments.utils.helper_functions import BNN_helper_functions
from fedgvi_experiments.utils.training_utils import EarlyStopping

# =============================================================================
# Classification through Bayesian Neural Network with FedGVI
#
# This code base was adapted by the authors (Mildner et al., 2025)
# from Ashman et al. (2022), their code is publicly available at:
# https://github.com/MattAshman/pvi and is licensed under an MIT license
# allowing us to adapt their code to FedGVI and freely distribute this.
# =============================================================================

seeds = [42, 676,  93, 215, 318, 242]

parser = argparse.ArgumentParser()
 
parser.add_argument("--run", type=int, default=1)
parser.add_argument("--run_all", type=bool, default=False)
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--client_split", type=str, default="homogeneous")
parser.add_argument("--loss", type=str, default="nll")
parser.add_argument("--loss_param", type=float, default=1.0)
parser.add_argument("--server_div", type=str, default="KLD")
parser.add_argument("--server_div_param", type=float, default=1.0)
parser.add_argument("--client_div", type=str, default="KLD")
parser.add_argument("--client_div_param", type=float, default=1.0)
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--server_iters", type=int, default=25)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--contamination_rate", type=float, default=0.1)
parser.add_argument("--contamination_type_random", type=bool, default=False)
parser.add_argument("--contamination_seed", type=float, default=42)
parser.add_argument("--optimiser", type=str, default="Adam")
parser.add_argument("--optimiser_epochs", type=int, default=2500)
parser.add_argument("--unif_prior", type=bool, default=False)
parser.add_argument("--net_type", type=str, default="fc1h")
parser.add_argument("--output_dir", type=str, default="./outputs")

args = parser.parse_args()
print(args)

run = args.run
run_all = args.run_all
dataset = args.dataset
num_clients = args.num_clients
client_split = args.client_split
loss = args.loss
loss_param = args.loss_param
server_div = args.server_div
server_div_param = args.server_div_param
client_div = args.client_div
client_div_param = args.client_div_param
learning_rate = args.learning_rate
server_iters = args.server_iters
device = args.device
contamination_rate = args.contamination_rate
contamination_type_random = args.contamination_type_random
contamination_seed = args.contamination_seed
optimiser = args.optimiser
optimiser_epochs = args.optimiser_epochs
unif_prior = args.unif_prior
net_type = args.net_type
output_dir = Path(args.output_dir)

if loss == "gce": #These two names are equivalent and we defer to rcce in the BNNs for convenience
    loss = "rcce"

damping = 1 / num_clients
method = 'pvi' if (client_div == "KLD" and server_div == "KLD" and loss == "nll") else "fedgvi"
if num_clients == 1:
    method = 'vi' if (method == 'pvi') else 'gvi'

"""print(run, run_all, dataset, num_clients, client_split, loss, loss_param, server_div, server_div_param, client_div, client_div_param,\
    client_div, client_div_param, learning_rate, server_iters, device, contamination_rate, contamination_type_random, contamination_seed,\
    optimiser, optimiser_epochs)"""

if client_div == "AR" and client_div_param == 1.0:
    print(f"Client Divergence as {client_div} specified with parameter {client_div_param}")
    client_div = "KLD"
    print("New client divergence: ", client_div)

if client_div == "AR" and client_div_param == 0.0:
    print(f"Client Divergence as {client_div} specified with parameter {client_div_param}")
    client_div = "RKL"
    print("New client divergence: ", client_div)

if loss == "rcce" and loss_param == 0.0:
    print(f"Client Loss as {loss} specified with parameter {loss_param}")
    loss = "nll"
    print("New client loss: ", loss)



if dataset == "MNIST":
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_test)

    train_data = {
        "x": ((train_set.data - 0) / 255).reshape(-1, 28 * 28),
        "y": train_set.targets,
    }

    test_data = {
        "x": ((test_set.data - 0) / 255).reshape(-1, 28 * 28),
        "y": test_set.targets,
    }
    num_labels = 10
elif dataset == "KMNIST":
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.KMNIST(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.KMNIST(root=data_dir, train=False, download=True, transform=transform_test)

    train_data = {
        "x": ((train_set.data - 0) / 255).reshape(-1, 28 * 28),
        "y": train_set.targets,
    }

    test_data = {
        "x": ((test_set.data - 0) / 255).reshape(-1, 28 * 28),
        "y": test_set.targets,
    }
    num_labels = 10
elif dataset == "FashionMNIST":
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_test)

    train_data = {
        "x": ((train_set.data - 0) / 255).reshape(-1, 28 * 28),
        "y": train_set.targets,
    }

    test_data = {
        "x": ((test_set.data - 0) / 255).reshape(-1, 28 * 28),
        "y": test_set.targets,
    }
    num_labels = 10

elif dataset == "Kuzushiji-49":
    
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    # The code to download the data is taken from: https://github.com/rois-codh/kmnist/blob/master/download_data.py
    # Check if data set is downloaded
    data_k49 = Path("../data/k49/k49-train-imgs.npz")
    if data_k49.is_file() == False:
        #Download data set if not existing
        import requests
        from tqdm import tqdm
        print("Downloading Data")

        url_list = ['http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz']
        
        for url in url_list:
            path = "../data/k49/" + url.split('/')[-1]
            r = requests.get(url, stream=True)
            with open(path, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

                for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                    if chunk:
                        f.write(chunk)
        print('All dataset files downloaded!')
    
    train_set = torch.from_numpy(np.load("../data/k49/k49-train-imgs.npz")['arr_0'])
    train_labels = torch.from_numpy(np.load("../data/k49/k49-train-labels.npz")['arr_0'])
    test_set = torch.from_numpy(np.load("../data/k49/k49-test-imgs.npz")['arr_0'])
    test_labels = torch.from_numpy(np.load("../data/k49/k49-test-labels.npz")['arr_0'])

    train_data = {
        "x": ((train_set - 0) / 255).reshape(-1, 28 * 28),
        "y": train_labels,
    }

    test_data = {
        "x": ((test_set- 0) / 255).reshape(-1, 28 * 28),
        "y": test_labels,
    }
    num_labels = 49
    

else:
    print("Data set specified not Implemented")
    raise NotImplementedError

if contamination_rate > 0.0:
    train_data_true = copy.deepcopy(train_data)
    contaminated_indices = BNN_helper_functions.contaminate_labels(train_data, contamination_rate, random=contamination_type_random, seed=contamination_seed)
    for i in contaminated_indices:
        assert train_data["y"][i] != train_data_true["y"][i]
else:
    train_data_true = copy.deepcopy(train_data)
    contaminated_indices = []

if client_split == "homogeneous":
    client_data = BNN_helper_functions.homogeneous_split(train_data, num_clients, seed=seeds[run])
elif client_split == "heterogeneous_80_20":

    client_data = BNN_helper_functions.heterogeneous_split_80_20(train_data, num_labels, num_clients, seed=seeds[run])
elif client_split == "heterogeneous_amounts":
    client_data = BNN_helper_functions.heterogeneous_split_random(train_data, num_clients, seed=seeds[run])
else:
    print("Client data split not formalised properly.")
    raise NotImplementedError

def performance_metrics(client, data, batch_size=512):
    dataset = torch.utils.data.TensorDataset(data["x"], data["y"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)
    
    device = client.config["device"]
    
    if device == "cuda:0" or device == "cuda:1":
        loader.pin_memory = True
        
    preds, nlls = [], []
    for (x_batch, y_batch) in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        pp = client.model_predict(x_batch)
        preds.append(pp.component_distribution.probs.mean(1).cpu())
        nlls.append(pp.log_prob(y_batch).cpu())
        
    nll = torch.cat(nlls).mean()
    preds = torch.cat(preds)
    acc = sum(torch.argmax(preds, dim=-1) == loader.dataset.tensors[1]) / len(
        loader.dataset.tensors[1]
    )
    
    return {"nll": nll, "acc": acc}

def transform_performance_metrics(metrics):
    ret = {}
    for k in metrics[0].keys():
        ret[k] = []
    for i in range(len(metrics)):
        for k in metrics[i].keys():
            ret[k].append(metrics[i][k])
    return ret

def FedGVI():
    if net_type == "fc1h": 
        print("Net Type: 1 Hidden Layer")
        dim_hidden = 200
        num_layers = 1
    elif net_type == "fc":
        print("Net Type: 2 Hidden Layers")
        dim_hidden = 100
        num_layers = 2
    else:
        print(f"Net Type {net_type} not yet supported. Choose one of ['fc1h','fc'].")
        raise NotImplementedError
    
    model_config = {
        "input_dim": train_data["x"][0].shape[0],
        "latent_dim": dim_hidden,
        "output_dim": num_labels,
        "num_layers": num_layers,
        "num_predictive_samples": 200,
        "prior_var": 1.0,
    }
    #print(model_config)

    client_config = {
        "damping_factor": damping,
        "optimiser": optimiser,
        "optimiser_params": {"lr": learning_rate},
        "sigma_optimiser_params": {"lr": 5e-4},
        "early_stopping": EarlyStopping(10, score_name="elbo", stash_model=True),
        "performance_metrics": performance_metrics,
        "batch_size": 256,
        "epochs": optimiser_epochs,
        "print_epochs": optimiser_epochs-1,
        "num_elbo_samples": 10,
        "valid_factors": False,
        "device": device,
        "init_var": 1e-3,
        "verbose": True,
        "divergence": client_div,
        "alpha": client_div_param,
        "loss": loss,
        "loss_param": loss_param,
    }

    server_config = {
        **client_config,
        #100
        "max_iterations": server_iters,
    }

    model = ClassificationBNNLocalRepam(config=model_config)

    # Initial parameters.
    if unif_prior:
        init_q_std_params = {
            "loc": torch.zeros(size=(model.num_parameters,)).to(device).uniform_(-0.1, 0.1),
            "scale": torch.ones(size=(model.num_parameters,)).to(device) 
            * client_config["init_var"] ** 0.5,
        }
    else:
        init_q_std_params = {
            "loc": torch.zeros(size=(model.num_parameters,)).to(device),
            "scale": torch.ones(size=(model.num_parameters,)).to(device) 
            * client_config["init_var"] ** 0.5,
        }

    prior_std_params = {
        "loc": torch.zeros(size=(model.num_parameters,)).to(device),
        "scale": model_config["prior_var"] ** 0.5 
        * torch.ones(size=(model.num_parameters,)).to(device),
    }

    init_factor_nat_params = {
        "np1": torch.zeros(model.num_parameters).to(device),
        "np2": torch.zeros(model.num_parameters).to(device),
    }

    p = MeanFieldGaussianDistribution(
        std_params=prior_std_params, is_trainable=False
    )
    init_q = MeanFieldGaussianDistribution(
        std_params=init_q_std_params, is_trainable=False
    )

    clients = []
    size_training = 0
    for i in range(num_clients):
        data_i = client_data[i]
        size_training += len(data_i["x"])
        t_i = MeanFieldGaussianFactor(nat_params=init_factor_nat_params)
        clients.append(
            Client(
                data=data_i,
                model=model,
                t=t_i,
                config=client_config,
                val_data=test_data
            )
        )    
    server = FedGVIServer(model=model, p=p, clients=clients, config=server_config, init_q=init_q, data=train_data, val_data=test_data)
    
    i = 1
    while not server.should_stop():
        server.tick()

        # Obtain performance metrics.
        metrics = server.log["performance_metrics"][-1]
        print("\n Iterations: {}.".format(i))
        print("Time taken: {:.3f}.".format(metrics["time"]))
        print(
            "Test nll: {:.3f}. Test acc: {:.3f}.".format(
                metrics["val_nll"], metrics["val_acc"]
            )
        )
        print(
            "Train nll: {:.3f}. Train acc: {:.3f}.\n".format(
                metrics["train_nll"], metrics["train_acc"]
            )
        )
        i += 1
    
    return transform_performance_metrics(server.log["performance_metrics"])

if __name__ == "__main__":
    outputs_fedgvi = FedGVI()
    if dataset == "MNIST":
        output_file_name = f"{method}_{num_clients}-{client_split}-Clients_{str(contamination_rate)}-contamination_{client_div}_{loss}_run_{str(run)}.txt" 
    else:
        output_file_name = f"{method}_{num_clients}-{client_split}-Clients_{dataset}_{str(contamination_rate)}-contamination_{client_div}_{loss}_run_{str(run)}.txt" 
    name = os.path.join(output_dir,output_file_name)
    
    file1 = open(name,"a")
    file1.write(f"{method} run {run} with: \n\
                - Number of Clients: {num_clients} ({client_split} split)\n\
                - Client divergence: {client_div}({client_div_param}) \n\
                - Server divergence: {server_div}({server_div_param}) \n\
                - Client loss: {loss} ({loss_param})\n\
                - Random contamination: {contamination_type_random} ({contamination_rate}, {contamination_seed})\n\
                - Optimiser: {optimiser} (lr: {learning_rate}, epochs: {optimiser_epochs}, server iters: {server_iters})\n\n\
                RESULTS:\n")
    for k in outputs_fedgvi.keys():
        file1.write(f"{k}: {outputs_fedgvi[k]}\n")
    file1.write("\n==========================================================\n")
    file1.close()
    print("Written to File")