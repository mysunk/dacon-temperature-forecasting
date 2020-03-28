#!/bin/sh
python3 tuning.py --max_evals=500 --save_file=Y16 --N_T=12 --N_S=20 --method=svr --nfold=30 --label=Y16 
python3 tuning.py --max_evals=500 --save_file=Y15 --N_T=12 --N_S=20 --method=svr --nfold=30 --label=Y15
python3 tuning.py --max_evals=500 --save_file=Y09 --N_T=12 --N_S=20 --method=svr --nfold=30 --label=Y09
