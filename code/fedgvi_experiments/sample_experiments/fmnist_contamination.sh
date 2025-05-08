#!/bin/sh
cd ..
for run in 1 2 3 4 5
do
echo "Running FedGVI with GCE delta=1"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.0 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss gce --loss_param 1.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.1 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss gce --loss_param 1.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.2 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss gce --loss_param 1.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.4 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss gce --loss_param 1.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.6 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss gce --loss_param 1.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
done

for run in 1 2 3 4 5
do
echo "Running PVI"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.0 --dataset FashionMNIST --client_div KLD --client_div_param 2.5 --loss nll --loss_param 0.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.1 --dataset FashionMNIST --client_div KLD --client_div_param 2.5 --loss nll --loss_param 0.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.2 --dataset FashionMNIST --client_div KLD --client_div_param 2.5 --loss nll --loss_param 0.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.4 --dataset FashionMNIST --client_div KLD --client_div_param 2.5 --loss nll --loss_param 0.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.6 --dataset FashionMNIST --client_div KLD --client_div_param 2.5 --loss nll --loss_param 0.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
done

for run in 1 2 3 4 5
do
echo "Running FedGVI with GCE delta=0.5"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.0 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss gce --loss_param 0.5 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.1 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss gce --loss_param 0.5 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.2 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss gce --loss_param 0.5 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.4 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss gce --loss_param 0.5 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.6 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss gce --loss_param 0.5 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"

echo "Running FedGVI with GCE delta=0"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.0 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss nll --loss_param 0.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.1 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss nll --loss_param 0.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.2 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss nll --loss_param 0.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.4 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss nll --loss_param 0.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
python3 run_fedgvi.py --run $run --device cuda:0 --num_clients 3 --contamination_rate 0.6 --dataset FashionMNIST --client_div AR --client_div_param 2.5 --loss nll --loss_param 0.0 --server_iters 25 --contamination_type_random True --net_type fc --output_dir "./results_contamin"
done