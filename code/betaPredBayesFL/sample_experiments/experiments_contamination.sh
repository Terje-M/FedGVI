#!/bin/sh
echo "RUN EXPERIMENTS ON FASHION-MNIST WITH DIFFERENT RATES OF CONTAMINATION"
cd ..
for seed in 676 93 215 318 242
do
    echo "Federated Averaging"
    python3 run_exp.py --mode "fed_sgd" --dataset f_mnist --epoch_per_client 25 --lr 1e-1 --num_round 25 --optim_type "sgdm" --seed $seed --save_dir "./results/Zero/" --device cuda:1 --num_clients 3 --contamination_rate 0.0 --contamination_type_random True
    python3 run_exp.py --mode "fed_sgd" --dataset f_mnist --epoch_per_client 25 --lr 1e-1 --num_round 25 --optim_type "sgdm" --seed $seed --save_dir "./results/Ten/" --device cuda:1 --num_clients 3 --contamination_rate 0.1 --contamination_type_random True
    python3 run_exp.py --mode "fed_sgd" --dataset f_mnist --epoch_per_client 25 --lr 1e-1 --num_round 25 --optim_type "sgdm" --seed $seed --save_dir "./results/Twenty/" --device cuda:1 --num_clients 3 --contamination_rate 0.2 --contamination_type_random True
    python3 run_exp.py --mode "fed_sgd" --dataset f_mnist --epoch_per_client 25 --lr 1e-1 --num_round 25 --optim_type "sgdm" --seed $seed --save_dir "./results/Fourty/" --device cuda:1 --num_clients 3 --contamination_rate 0.4 --contamination_type_random True
    echo "Federated Posterior Averaging"
    python3 run_exp.py --mode "fed_pa" --dataset f_mnist --epoch_per_client 25 --lr 1e-2 --num_round 25 --optim_type "sgdm" --seed $seed --save_dir "./results/Zero/" --rho 0.4 --g_lr 5e-1 --device cuda:1 --num_clients 3 --contamination_rate 0.0 --contamination_type_random True
    python3 run_exp.py --mode "fed_pa" --dataset f_mnist --epoch_per_client 25 --lr 1e-2 --num_round 25 --optim_type "sgdm" --seed $seed --save_dir "./results/Ten/" --rho 0.4 --g_lr 5e-1 --device cuda:1 --num_clients 3 --contamination_rate 0.1 --contamination_type_random True
    python3 run_exp.py --mode "fed_pa" --dataset f_mnist --epoch_per_client 25 --lr 1e-2 --num_round 25 --optim_type "sgdm" --seed $seed --save_dir "./results/Twenty/" --rho 0.4 --g_lr 5e-1 --device cuda:1 --num_clients 3 --contamination_rate 0.2 --contamination_type_random True
    python3 run_exp.py --mode "fed_pa" --dataset f_mnist --epoch_per_client 25 --lr 1e-2 --num_round 25 --optim_type "sgdm" --seed $seed --save_dir "./results/Fourty/" --rho 0.4 --g_lr 5e-1 --device cuda:1 --num_clients 3 --contamination_rate 0.4 --contamination_type_random True
    echo "beta-PredBayes"
    python3 run_exp.py --mode "tune_distill_f_mcmc" --dataset f_mnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --seed $seed --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "adam" --kd_epochs 100 --save_dir "./results/Zero/" --device cuda:1 --num_clients 3 --contamination_rate 0.0 --contamination_type_random True
    python3 run_exp.py --mode "tune_distill_f_mcmc" --dataset f_mnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --seed $seed --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "adam" --kd_epochs 100 --save_dir "./results/Ten/" --device cuda:1 --num_clients 3 --contamination_rate 0.1 --contamination_type_random True
    python3 run_exp.py --mode "tune_distill_f_mcmc" --dataset f_mnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --seed $seed --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "adam" --kd_epochs 100 --save_dir "./results/Twenty/" --device cuda:1 --num_clients 3 --contamination_rate 0.2 --contamination_type_random True
    python3 run_exp.py --mode "tune_distill_f_mcmc" --dataset f_mnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --seed $seed --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "adam" --kd_epochs 100 --save_dir "./results/Fourty/" --device cuda:1 --num_clients 3 --contamination_rate 0.4 --contamination_type_random True
done