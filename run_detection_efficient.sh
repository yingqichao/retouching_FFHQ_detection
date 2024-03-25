python -m torch.distributed.launch --master_port 11307 --nproc_per_node=1 train.py \
          -opt options/meiyan_efficient.yml -mode 0 -task_name baseline_efficient_512 \
          --launcher pytorch