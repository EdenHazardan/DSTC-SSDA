cd /home/gaoy/CCDLD-SSDA && \
python3 exp/warm-up/target_100_only/python/train.py \
        --exp_name target_100_only \
        --weight_res101 /home/gaoy/CCDLD-SSDA/pretrained/DeepLab_resnet_pretrained_init-f81d91e8.pth \
        --lr 2.5e-4 \
        --distance 2 \
        --source_batch_size 2 \
        --target_batch_size 2 \
        --train_num_workers 2 \
        --test_batch_size 1 \
        --test_num_workers 2 \
        --early_stop 120000 \
        --train_iterations 250000 \
        --log_interval 100 \
        --val_interval 2000 \
        --work_dirs /home/gaoy/CCDLD-SSDA/work_dirs/warm-up \