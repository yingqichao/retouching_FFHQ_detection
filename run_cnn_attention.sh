python -m torch.distributed.launch  --master_port 9992 --nproc_per_node=1 train.py -opt \
options/dual/meiyan_cnn_attention.yml -mode 0 --launcher pytorch