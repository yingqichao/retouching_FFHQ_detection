python -m torch.distributed.launch  --master_port 4679 \
       --nproc_per_node=1 train.py -opt options/others/calc_psnr.yml -mode 1 \
       --launcher pytorch