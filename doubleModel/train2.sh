
seed=1
#lr='2e-4'
lr='2e-7'  # 从2e-6调整为2e-7
batch_size=16
epoch=10
input_steps=1
predict_steps=1
max_t=6
input_var_list='so thetao tos uo vo zos'

save_eval_steps=100

dist_port=$[12345+$[$RANDOM%12345]]

output_dir=/mnt/data/zhu.yishun/ORCA-DL-main/double_output2e-7 # configure your output directory
data_dir=/mnt/data/zhu.yishun/ORCA-DL-main/data/train_data # replace with your CMIP data directory, e.g., ./download/train_data/
soda_dir=/mnt/data/zhu.yishun/ORCA-DL-main/data/valid_test_data/SODA2 # replace with your SODA data directory, e.g., ./download/valid_test_data/SODA2
oras5_dir=/mnt/data/zhu.yishun/ORCA-DL-main/data/valid_test_data/ORAS5 # replace with your ORAS5 data directory e.g., ./download/valid_test_data/ORAS5
pretrained_model_path="/mnt/data/zhu.yishun/ORCA-DL-main/ori-model"  # 请替换为实际路径

### If you use SLURM to launch the training script, you can use the following command:
# node_num=1
# gpu_per_node=4
# srun -p YOUR_PARTITION_NAME --ntasks-per-node=$gpu_per_node -N $node_num --gres=gpu:$gpu_per_node --async \
#     python -u train.py
# bash /mnt/data/zhu.yishun/ORCA-DL-main/doubleModel/train2.sh
### Otherwise, you can use torchrun to launch the training script

torchrun --nproc_per_node=2 \
    /mnt/data/zhu.yishun/ORCA-DL-main/doubleModel/train2.py \
        --in_chans 16 16 1 16 16 1 \
        --out_chans 16 16 1 16 16 1 \
        --max_t $max_t \
        --atmo_var_list tauu tauv \
        --atmo_dims 2 \
        --model_path $pretrained_model_path \
        --ignore_mismatched_sizes True \
        --do_train \
        --dist_port $dist_port \
        --data_dir $data_dir \
        --input_var_list $input_var_list \
        --input_steps $input_steps \
        --predict_steps $predict_steps \
        --output_dir $output_dir \
        --seed $seed \
        --report_to tensorboard \
        --log_level info \
        --logging_dir $output_dir/log \
        --logging_steps 30 \
        --log_on_each_node False \
        --save_strategy steps \
        --save_steps $save_eval_steps \
        --save_total_limit 3 \
        --ddp_find_unused_parameters True \
        --num_train_epochs $epoch \
        --per_device_train_batch_size $batch_size \
        --per_device_eval_batch_size $batch_size \
        --gradient_accumulation_steps 1 \
        --dataloader_num_workers 8 \
        --gradient_checkpointing True \
        --fp16 \
        --learning_rate $lr \
        --weight_decay 0.1 \
        --max_grad_norm 0.0 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --adam_epsilon 1e-6 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --do_eval \
        --valid_data_dir $soda_dir $oras5_dir \
        --end_year 2010 \
        --evaluation_strategy steps \
        --eval_steps $save_eval_steps \
        --load_best_model_at_end True