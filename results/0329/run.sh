#!/bin/sh
python3 tuning.py --max_evals=1000 --save_file=0329/Y10_lgb --N_T=12 --N_S=20 --method=lgb --nfold=30 --label=Y10
python3 tuning.py --max_evals=200 --save_file=0329/Y10_svr --N_T=12 --N_S=20 --method=svr --nfold=30 --label=Y10
python3 tuning.py --max_evals=200 --save_file=0329/Y10_rf --N_T=12 --N_S=20 --method=rf --nfold=30 --label=Y10
python3 tuning.py --max_evals=1000 --save_file=0329/Y11_lgb --N_T=12 --N_S=20 --method=lgb --nfold=30 --label=Y11
python3 tuning.py --max_evals=200 --save_file=0329/Y11_svr --N_T=12 --N_S=20 --method=svr --nfold=30 --label=Y11
python3 tuning.py --max_evals=200 --save_file=0329/Y11_rf --N_T=12 --N_S=20 --method=rf --nfold=30 --label=Y11
python3 tuning.py --max_evals=1000 --save_file=0329/Y12_lgb --N_T=12 --N_S=20 --method=lgb --nfold=30 --label=Y12
python3 tuning.py --max_evals=200 --save_file=0329/Y12_svr --N_T=12 --N_S=20 --method=svr --nfold=30 --label=Y12
python3 tuning.py --max_evals=200 --save_file=0329/Y12_rf --N_T=12 --N_S=20 --method=rf --nfold=30 --label=Y12
python3 tuning.py --max_evals=1000 --save_file=0329/Y13_lgb --N_T=12 --N_S=20 --method=lgb --nfold=30 --label=Y13
python3 tuning.py --max_evals=200 --save_file=0329/Y13_svr --N_T=12 --N_S=20 --method=svr --nfold=30 --label=Y13
python3 tuning.py --max_evals=200 --save_file=0329/Y13_rf --N_T=12 --N_S=20 --method=rf --nfold=30 --label=Y13
