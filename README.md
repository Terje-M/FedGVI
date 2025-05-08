# Federated Generalised Variational Inference
---
We present here the code base to reproduce the experiments in our paper [Federated Generalised Variational Inference: A Robust Probabilistic Federated Learning Framework](https://arxiv.org/abs/2502.00846). 

## Experiments

We have carried out three types of experiments: Synthetics, Logistic Regression, and Real-life classification on Bayesian Neural Networks.
All code has been written in Python using PyTorch. The full requirements can be found in the requirements.txt file.
Data will either be created through simulation with a fixed seed, or will be downloaded automatically when running the code, so no additional downloads are required.


### Sythetics and Logistic Regression

We have compiled the synthetics and logistic regression experiments into jupyter notebook files in which the experiments and the figures from the paper can be reproduced. 
These experiments were executed on a standard laptop CPU, and we suggest simply executing the entire jupyter notebook or individual cells as desired. 
We indicate where these experiments are more comutationally demanding and provide saved tensors for quick execution, in addition to the code used to produce these.
For the competing methods in logistic regression, we have borrowed the code from Kassab and Simeone (2022).

### Bayesian Neural Networks

These experiments were executed on an NVIDIA GeForce RTX 5090 GPU, and are generally computationally demanding on standard laptops, especially when running repetitions and different initialisations.
We provide the full code to reproduce the experiments as well as sample shell scipts that can be used to replicate the experiments.

### Running Experiments

#### Synthetics

Navigating to the Jupyter Notebook file `code/Notebooks/Toy_Experiments.ipynb' and running the experiments through this, as we generate synthetic data and figures in this environment.

#### Logistic Regression

We suggest navigating to the Jupyter Notebook file `code/Notebooks/Logistic\ Regression.ipynb' and running the experiments through this.

#### Bayesian Neural Networks

We provide sample shell scripts that allow you to run the experiments in `code/betaPredBayesFL/sample_experiments` and `code/fedgvi_experiments/sample_experiments` respectively for the competing methods (FedAvg, FedPA, and $\beta$-PredBayes) and FedGVI (including VI, PVI, and GVI).

To customise the experiments, please use the argument flags in `code/betaPredBayesFL/run_exp.py` and `code/fedgvi_experiments/run_fedgvi.py`. 

For example, we have for FedGVI:

```
python3 run_fedgvi.py --run 1 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:0 --contamination_rate 0.2
```

or for e.g. $\beta$-PredBayes:
```
python3 run_exp.py --mode "tune_distill_f_mcmc" --dataset f_mnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --seed $seed --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "adam" --kd_epochs 100 --device cuda:0 --num_clients 10 --contamination_rate 0.2

```

## Citation

```
@inproceedings{mildner2025,
      title={Federated Generalised Variational Inference: A Robust Probabilistic Federated Learning Framework},
      author={Terje Mildner and Oliver Hamelijnck and Paris Giampouras and Theodoros Damoulas},
      booktitle={Forty-second International Conference on Machine Learning},
      year={2025},
      url={https://openreview.net/forum?id=M7mVzCV6uU}
}
```

## Installation
```bash
conda create -n fedgvi python=3.10
conda activate fedgvi
pip install -r requirements.txt
conda install ipykernel nb_conda_kernels jupyter # notebooks
```
