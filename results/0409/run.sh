#!/bin/sh
python3 tuning.py --max_evals=1000 --save_file=0409/Y16_lgb --method=lgb --nfold=10 --label=Y16
python3 tuning.py --max_evals=1000 --save_file=0409/Y11_lgb --method=lgb --nfold=10 --label=Y11
python3 tuning.py --max_evals=1000 --save_file=0409/Y17_lgb --method=lgb --nfold=10 --label=Y17
python3 tuning.py --max_evals=1000 --save_file=0409/Y09_lgb --method=lgb --nfold=10 --label=Y09
