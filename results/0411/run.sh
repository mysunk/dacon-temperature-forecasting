#!/bin/sh
python3 tuning_residual.py --max_evals=3000 --save_file=0411/Y16_residual_rf --method=rf --nfold=10 --label=Y16
python3 tuning_residual.py --max_evals=3000 --save_file=0411/Y16_residual_lgb --method=lgb --nfold=10 --label=Y16
python3 tuning_residual.py --max_evals=3000 --save_file=0411/Y17_residual_rf --method=rf --nfold=10 --label=Y17
python3 tuning_residual.py --max_evals=3000 --save_file=0411/Y17_residual_lgb --method=lgb --nfold=10 --label=Y17
python3 tuning_residual.py --max_evals=3000 --save_file=0411/Y09_residual_rf --method=rf --nfold=10 --label=Y09
python3 tuning_residual.py --max_evals=3000 --save_file=0411/Y09_residual_lgb --method=lgb --nfold=10 --label=Y09
python3 tuning_residual.py --max_evals=3000 --save_file=0411/Y11_residual_rf --method=rf --nfold=10 --label=Y11
python3 tuning_residual.py --max_evals=3000 --save_file=0411/Y11_residual_lgb --method=lgb --nfold=10 --label=Y11
