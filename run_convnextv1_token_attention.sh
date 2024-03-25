python -m torch.distributed.launch --master_port 33501 --nproc_per_node=1 train.py \
          -opt options/plug_in/meiyan_convnextv1_token_attention.yml -mode 0 \
          --launcher pytorch