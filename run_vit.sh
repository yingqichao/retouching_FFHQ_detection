python -m torch.distributed.launch  --master_port 3777 --nproc_per_node=1 train.py -opt \
options/dual/meiyan_vit_dual.yml -mode 0 --launcher pytorch