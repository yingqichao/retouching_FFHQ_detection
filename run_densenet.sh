python -m torch.distributed.launch  --master_port 10005 --nproc_per_node=1 train.py -opt \
options/dual/meiyan_densenet_dual.yml -mode 0 --launcher pytorch