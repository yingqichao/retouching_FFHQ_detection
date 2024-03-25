python -m torch.distributed.launch --master_port 41111 --nproc_per_node=1 train.py \
          -opt options/plug_in/meiyan_convnextv1_cnn_attention.yml -mode 0 \
          --launcher pytorch