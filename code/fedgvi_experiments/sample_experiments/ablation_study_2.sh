#!/bin/sh
cd ..
echo "Run ablation study on Learning Rate of FedGVI."
for run in 1 2 3 4 5
do
    for lr in 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5
    do
        echo "Running FedGVI with divergence D_AR for learning rate = $lr"
        python3 run_fedgvi.py --run $run --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:0 --num_clients 3 --learning_rate $lr
        python3 run_fedgvi.py --run $run --loss gce --loss_param 0.0 --client_div AR --client_div_param 2.5 --device cuda:0 --num_clients 3 --learning_rate $lr
        python3 run_fedgvi.py --run $run --loss gce --loss_param 0.8 --client_div AR --client_div_param 1.0 --device cuda:0 --num_clients 3 --learning_rate $lr
        python3 run_fedgvi.py --run $run --loss gce --loss_param 0.0 --client_div AR --client_div_param 1.0 --device cuda:0 --num_clients 3 --learning_rate $lr
    done
done