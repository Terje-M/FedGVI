#!/bin/sh
cd ..
echo "=============================\n 10 Clients \n ============================="
echo "\n P(NLL, KLD) \n"
python3 run_fedgvi.py --run 1 --device cuda:0
python3 run_fedgvi.py --run 2 --device cuda:0
python3 run_fedgvi.py --run 3 --device cuda:0
python3 run_fedgvi.py --run 4 --device cuda:0
python3 run_fedgvi.py --run 5 --device cuda:0
echo "\n P(Gen_CE, AR)\n"
python3 run_fedgvi.py --run 1 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:0
python3 run_fedgvi.py --run 2 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:0
python3 run_fedgvi.py --run 3 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:0
python3 run_fedgvi.py --run 4 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:0
python3 run_fedgvi.py --run 5 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:0
echo "\n P(Gen_CE, KLD)\n"
python3 run_fedgvi.py --run 1 --loss gce --loss_param 0.8 --client_div KLD --client_div_param 2.5 --device cuda:0
python3 run_fedgvi.py --run 2 --loss gce --loss_param 0.8 --client_div KLD --client_div_param 2.5 --device cuda:0
python3 run_fedgvi.py --run 3 --loss gce --loss_param 0.8 --client_div KLD --client_div_param 2.5 --device cuda:0
python3 run_fedgvi.py --run 4 --loss gce --loss_param 0.8 --client_div KLD --client_div_param 2.5 --device cuda:0
python3 run_fedgvi.py --run 5 --loss gce --loss_param 0.8 --client_div KLD --client_div_param 2.5 --device cuda:0
echo "\n P(NLL, AR)\n"
python3 run_fedgvi.py --run 1 --client_div AR --client_div_param 2.5 --device cuda:1
python3 run_fedgvi.py --run 2 --client_div AR --client_div_param 2.5 --device cuda:1
python3 run_fedgvi.py --run 3 --client_div AR --client_div_param 2.5 --device cuda:1
python3 run_fedgvi.py --run 4 --client_div AR --client_div_param 2.5 --device cuda:1
python3 run_fedgvi.py --run 5 --client_div AR --client_div_param 2.5 --device cuda:1
echo "=============================\n 3 Clients \n ============================="
echo "\n P(Gen_CE, AR)\n"
python3 run_fedgvi.py --run 1 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:1 --num_clients 3
python3 run_fedgvi.py --run 2 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:1 --num_clients 3
python3 run_fedgvi.py --run 3 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:1 --num_clients 3
python3 run_fedgvi.py --run 4 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:1 --num_clients 3
python3 run_fedgvi.py --run 5 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:1 --num_clients 3
echo "\n P(NLL, AR)\n"
python3 run_fedgvi.py --run 1 --client_div AR --client_div_param 2.5 --device cuda:0 --num_clients 3
python3 run_fedgvi.py --run 2 --client_div AR --client_div_param 2.5 --device cuda:0 --num_clients 3
python3 run_fedgvi.py --run 3 --client_div AR --client_div_param 2.5 --device cuda:0 --num_clients 3
python3 run_fedgvi.py --run 4 --client_div AR --client_div_param 2.5 --device cuda:0 --num_clients 3
python3 run_fedgvi.py --run 5 --client_div AR --client_div_param 2.5 --device cuda:0 --num_clients 3
echo "\n P(Gen_CE, KLD)\n"
python3 run_fedgvi.py --run 1 --loss gce --loss_param 0.8 --client_div KLD --client_div_param 2.5 --device cuda:1 --num_clients 3
python3 run_fedgvi.py --run 2 --loss gce --loss_param 0.8 --client_div KLD --client_div_param 2.5 --device cuda:1 --num_clients 3
python3 run_fedgvi.py --run 3 --loss gce --loss_param 0.8 --client_div KLD --client_div_param 2.5 --device cuda:1 --num_clients 3
python3 run_fedgvi.py --run 4 --loss gce --loss_param 0.8 --client_div KLD --client_div_param 2.5 --device cuda:1 --num_clients 3
python3 run_fedgvi.py --run 5 --loss gce --loss_param 0.8 --client_div KLD --client_div_param 2.5 --device cuda:1 --num_clients 3
echo "\n P(NLL, KLD) \n"
python3 run_fedgvi.py --run 1 --device cuda:1 --num_clients 3
python3 run_fedgvi.py --run 2 --device cuda:1 --num_clients 3
python3 run_fedgvi.py --run 3 --device cuda:1 --num_clients 3
python3 run_fedgvi.py --run 4 --device cuda:1 --num_clients 3
python3 run_fedgvi.py --run 5 --device cuda:1 --num_clients 3
echo "=============================\n 1 Client \n ============================="
echo "\n P(Gen_CE, AR)\n"
python3 run_fedgvi.py --run 1 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:1 --num_clients 1
python3 run_fedgvi.py --run 2 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:1 --num_clients 1
python3 run_fedgvi.py --run 3 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:1 --num_clients 1
python3 run_fedgvi.py --run 4 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:1 --num_clients 1
python3 run_fedgvi.py --run 5 --loss gce --loss_param 0.8 --client_div AR --client_div_param 2.5 --device cuda:1 --num_clients 1
echo "\n P(NLL, KLD) \n"
python3 run_fedgvi.py --run 1 --device cuda:1 --num_clients 1
python3 run_fedgvi.py --run 2 --device cuda:1 --num_clients 1
python3 run_fedgvi.py --run 3 --device cuda:1 --num_clients 1
python3 run_fedgvi.py --run 4 --device cuda:1 --num_clients 1
python3 run_fedgvi.py --run 5 --device cuda:1 --num_clients 1