python -m torch.distributed.launch  --master_port 31110 --nproc_per_node=1 train.py -opt \
options/dual/meiyan_swin_dual.yml -mode 0 --launcher pytorch