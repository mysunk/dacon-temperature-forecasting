#!/bin/sh
python3 tuning.py --max_evals=300 --save_file=0402/Y13_svr --method=svr --nfold=30 --label=Y13
python3 tuning.py --max_evals=3000 --save_file=0402/Y13_lgb --method=lgb --nfold=30 --label=Y13
python3 tuning.py --max_evals=300 --save_file=0402/Y13_rf --method=rf --nfold=30 --label=Y13
python3 tuning.py --max_evals=300 --save_file=0402/Y15_svr --method=svr --nfold=30 --label=Y15
python3 tuning.py --max_evals=3000 --save_file=0402/Y15_lgb --method=lgb --nfold=30 --label=Y15
python3 tuning.py --max_evals=300 --save_file=0402/Y15_rf --method=rf --nfold=30 --label=Y15
