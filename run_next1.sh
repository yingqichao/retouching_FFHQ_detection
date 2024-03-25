python -m torch.distributed.launch  --master_port 31220 --nproc_per_node=1 train.py -opt \
options/dual/meiyan_convnext_dual.yml -mode 0 --launcher pytorch