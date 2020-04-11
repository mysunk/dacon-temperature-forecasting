#!/bin/sh
# python3 tuning_residual.py --max_evals=10000 --save_file=0410/Y12_residual_rf --method=rf --nfold=10 --label=Y12
python3 tuning_residual.py --max_evals=10000 --save_file=0410/Y13_residual_rf --method=rf --nfold=10 --label=Y13
python3 tuning_residual.py --max_evals=10000 --save_file=0410/Y15_residual_rf --method=rf --nfold=10 --label=Y15
python3 tuning_residual.py --max_evals=10000 --save_file=0410/Y12_residual_lgb --method=lgb --nfold=10 --label=Y12
python3 tuning_residual.py --max_evals=10000 --save_file=0410/Y13_residual_lgb --method=lgb --nfold=10 --label=Y13
python3 tuning_residual.py --max_evals=10000 --save_file=0410/Y15_residual_lgb --method=lgb --nfold=10 --label=Y15
