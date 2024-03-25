python -m torch.distributed.launch  --master_port 22334 --nproc_per_node=1 train.py -opt \
options/dual/meiyan_PVT_dual.yml -mode 0 --launcher pytorch