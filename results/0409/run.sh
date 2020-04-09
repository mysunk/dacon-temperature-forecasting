#!/bin/sh
#python3 tuning.py --max_evals=3000 --save_file=0409/Y12_lgb --method=lgb --nfold=10 --label=Y12
python3 tuning.py --max_evals=1500 --save_file=0409/Y13_lgb --method=lgb --nfold=10 --label=Y13
python3 tuning.py --max_evals=1500 --save_file=0409/Y15_lgb --method=lgb --nfold=10 --label=Y15
python3 tuning.py --max_evals=3000 --save_file=0409/Y16_lgb --method=lgb --nfold=10 --label=Y16
