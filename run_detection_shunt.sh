python -m torch.distributed.launch --master_port 7777 --nproc_per_node=1 train.py \
          -opt options/meiyan_shunt.yml -mode 0 \
          --launcher pytorch