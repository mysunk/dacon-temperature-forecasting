#!/bin/sh
# python3 tuning.py --max_evals=1000 --save_file=0328/Y16 --N_T=12 --N_S=20 --method=rf --nfold=30 --label=Y16 
python3 tuning.py --max_evals=100 --save_file=0328/Y15 --N_T=12 --N_S=20 --method=rf --nfold=30 --label=Y15
python3 tuning.py --max_evals=100 --save_file=0328/Y09 --N_T=12 --N_S=20 --method=rf --nfold=30 --label=Y09
