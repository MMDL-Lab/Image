export PYTHONPATH=../:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python ../tools/train_val.py -e
