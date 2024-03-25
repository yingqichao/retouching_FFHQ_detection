python -m torch.distributed.launch  --master_port 9911 --nproc_per_node=1 train.py -opt \
options/dual/meiyan_shunt_token_attention.yml -mode 0 --launcher pytorch