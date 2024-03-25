python -m torch.distributed.launch  --master_port 33113 --nproc_per_node=1 train.py -opt \
options/dual/meiyan_convnextv2_dual.yml -mode 0 --launcher pytorch