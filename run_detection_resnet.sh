python -m torch.distributed.launch --master_port 2001 --nproc_per_node=1 train.py \
          -opt options/meiyan_resnet.yml -mode 0 -task_name baseline_resnet_512 \
          --launcher pytorch