#!/bin/sh
cd ..
echo "Run ablation study on Hyperparameters of FedGVI."
for run in 1 2 3 4 5
do
    for alpha in 0.0 0.5 1.0 1.5 2.5 5.0
    do
        for delta in 0.0 0.2 0.4 0.6 0.8 1.0
        do
            python3 run_fedgvi.py --run $run --loss gce --loss_param $delta --client_div AR --client_div_param $alpha --device cuda:0 --num_clients 5
        done
    done
done