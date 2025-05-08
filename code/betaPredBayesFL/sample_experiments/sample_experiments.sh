#!/bin/sh
echo "Run experiments on common architectures."
cd ..
for nettype in fc fc1h
do
    for numclients in 3 10
    do
        echo "Federated Averaging - $numclients Clients"
        python3 run_exp.py --mode "fed_sgd" --dataset mnist --epoch_per_client 10 --lr 1e-1 --num_round 25 --optim_type "sgdm" --seed 676 --save_dir "./results/" --device cuda:0 --net_type $nettype --num_clients $numclients
        python3 run_exp.py --mode "fed_sgd" --dataset mnist --epoch_per_client 10 --lr 1e-1 --num_round 25 --optim_type "sgdm" --seed 93 --save_dir "./results/" --device cuda:0 --net_type $nettype --num_clients $numclients
        python3 run_exp.py --mode "fed_sgd" --dataset mnist --epoch_per_client 10 --lr 1e-1 --num_round 25 --optim_type "sgdm" --seed 215 --save_dir "./results/" --device cuda:0 --net_type $nettype --num_clients $numclients
        python3 run_exp.py --mode "fed_sgd" --dataset mnist --epoch_per_client 10 --lr 1e-1 --num_round 25 --optim_type "sgdm" --seed 318 --save_dir "./results/" --device cuda:0 --net_type $nettype --num_clients $numclients
        python3 run_exp.py --mode "fed_sgd" --dataset mnist --epoch_per_client 10 --lr 1e-1 --num_round 25 --optim_type "sgdm" --seed 242 --save_dir "./results/" --device cuda:0 --net_type $nettype --num_clients $numclients
        echo "Federated Posterior Averaging - $numclients Clients"
        python3 run_exp.py --mode "fed_pa" --dataset mnist --epoch_per_client 10 --lr 1e-2 --num_round 25 --optim_type "sgdm" --seed 676 --save_dir "./results/" --rho 0.4 --g_lr 1 --device cuda:0 --net_type $nettype --num_clients $numclients
        python3 run_exp.py --mode "fed_pa" --dataset mnist --epoch_per_client 10 --lr 1e-2 --num_round 25 --optim_type "sgdm" --seed 93 --save_dir "./results/" --rho 0.4 --g_lr 1 --device cuda:0 --net_type $nettype --num_clients $numclients
        python3 run_exp.py --mode "fed_pa" --dataset mnist --epoch_per_client 10 --lr 1e-2 --num_round 25 --optim_type "sgdm" --seed 215 --save_dir "./results/" --rho 0.4 --g_lr 1 --device cuda:0 --net_type $nettype --num_clients $numclients
        python3 run_exp.py --mode "fed_pa" --dataset mnist --epoch_per_client 10 --lr 1e-2 --num_round 25 --optim_type "sgdm" --seed 318 --save_dir "./results/" --rho 0.4 --g_lr 1 --device cuda:0 --net_type $nettype --num_clients $numclients
        python3 run_exp.py --mode "fed_pa" --dataset mnist --epoch_per_client 10 --lr 1e-2 --num_round 25 --optim_type "sgdm" --seed 242 --save_dir "./results/" --rho 0.4 --g_lr 1 --device cuda:0 --net_type $nettype --num_clients $numclients
        echo "beta-PredBayes - $numclients Clients"
        python3 run_exp.py --mode "tune_distill_f_mcmc" --dataset mnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --seed 676 --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "sgdm" --kd_epochs 100 --save_dir "./results/" --device cuda:0 --net_type $nettype --num_clients $numclients
        python3 run_exp.py --mode "tune_distill_f_mcmc" --dataset mnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --seed 93 --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "sgdm" --kd_epochs 100 --save_dir "./results/" --device cuda:0 --net_type $nettype --num_clients $numclients
        python3 run_exp.py --mode "tune_distill_f_mcmc" --dataset mnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --seed 215 --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "sgdm" --kd_epochs 100 --save_dir "./results/" --device cuda:0 --net_type $nettype --num_clients $numclients
        python3 run_exp.py --mode "tune_distill_f_mcmc" --dataset mnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --seed 318 --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "sgdm" --kd_epochs 100 --save_dir "./results/" --device cuda:0 --net_type $nettype --num_clients $numclients
        python3 run_exp.py --mode "tune_distill_f_mcmc" --dataset mnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --seed 242 --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "sgdm" --kd_epochs 100 --save_dir "./results/" --device cuda:0 --net_type $nettype --num_clients $numclients
    done
done