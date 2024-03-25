python -m torch.distributed.launch --master_port 33499 --nproc_per_node=1 train.py \
          -opt options/plug_in/meiyan_resnet_token_attention.yml -mode 0 \
          --launcher pytorch