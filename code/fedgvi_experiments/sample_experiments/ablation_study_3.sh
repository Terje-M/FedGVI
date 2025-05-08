#!/bin/sh
cd ..
echo "Run ablation study on small changes in Hyperparameters of FedGVI."
for run in 1 2 3 4 5
do
    for delta in 0.75 0.775 0.79 0.8 0.81 0.825 0.85 
    do
        echo "Running FedGVI run = $run divergence D_AR for alpha = 2.5 and Loss GCE for delta = $delta"
        python3 run_fedgvi.py --run $run --loss gce --loss_param $delta --client_div AR --client_div_param 2.5 --device cuda:1 --num_clients 5 --output_dir "./ablation_study_3"
    done
done
