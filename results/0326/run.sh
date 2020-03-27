#!/bin/sh
python3 tuning.py --max_evals=500 --save_file=Y16_1 --N_T=12 --N_S=20 --method=lgb --nfold=30 --label=Y16 
python3 tuning.py --max_evals=500 --save_file=Y15_1 --N_T=12 --N_S=20 --method=lgb --nfold=30 --label=Y15
python3 tuning.py --max_evals=500 --save_file=Y09_1 --N_T=12 --N_S=20 --method=lgb --nfold=30 --label=Y09
python3 tuning.py --max_evals=500 --save_file=Y10_1 --N_T=12 --N_S=20 --method=lgb --nfold=30 --label=Y10
python3 tuning.py --max_evals=500 --save_file=Y11_1 --N_T=12 --N_S=20 --method=lgb --nfold=30 --label=Y11
