#!/bin/sh
python3 tuning.py --max_evals=200 --save_file=0330/Y14_svr --N_T=12 --N_S=20 --method=svr --nfold=30 --label=Y14
python3 tuning.py --max_evals=1000 --save_file=0330/Y14_lgb --N_T=12 --N_S=20 --method=lgb --nfold=30 --label=Y14
python3 tuning.py --max_evals=200 --save_file=0330/Y14_rf --N_T=12 --N_S=20 --method=rf --nfold=30 --label=Y14
