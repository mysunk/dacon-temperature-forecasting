#!/bin/sh
# python3 tuning.py --max_evals=300 --save_file=0404/Y16_svr --method=svr --nfold=30 --label=Y16
# python3 tuning.py --max_evals=3000 --save_file=0404/Y16_lgb --method=lgb --nfold=30 --label=Y16
python3 tuning.py --max_evals=300 --save_file=0404/Y16_rf --method=rf --nfold=30 --label=Y16
python3 tuning.py --max_evals=300 --save_file=0404/Y09_svr --method=svr --nfold=30 --label=Y09
python3 tuning.py --max_evals=3000 --save_file=0404/Y09_lgb --method=lgb --nfold=30 --label=Y09
python3 tuning.py --max_evals=300 --save_file=0404/Y09_rf --method=rf --nfold=30 --label=Y09
python3 tuning.py --max_evals=300 --save_file=0404/Y01_svr --method=svr --nfold=30 --label=Y01
python3 tuning.py --max_evals=3000 --save_file=0404/Y01_lgb --method=lgb --nfold=30 --label=Y01
python3 tuning.py --max_evals=300 --save_file=0404/Y01_rf --method=rf --nfold=30 --label=Y01
python3 tuning.py --max_evals=300 --save_file=0404/Y02_svr --method=svr --nfold=30 --label=Y02
python3 tuning.py --max_evals=3000 --save_file=0404/Y02_lgb --method=lgb --nfold=30 --label=Y02
python3 tuning.py --max_evals=300 --save_file=0404/Y02_rf --method=rf --nfold=30 --label=Y02
