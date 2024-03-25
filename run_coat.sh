python -m torch.distributed.launch --master_port 36328 --nproc_per_node=1 train.py \
          -opt options/dual/meiyan_coatnet_dual.yml -mode 0 \
          --launcher pytorch