
output_dir=./output/test/exp1
input_var_list='so thetao tos uo vo zos'  
predict_steps=24
input_steps=1

data_dir=YOUR_GODAS_DATA_DIR # replace with your GODAS data directory, e.g., ./download/valid_test_data/GODAS

ckpt_dir=YOUR_CKPT_DIR # replace with your checkpoint directory, it should contain the trained model weights, e.g., seed1.bin, pytorch_model.bin, etc.

batch_size=8

dist_port=$[31234+$[$RANDOM%100]]

python -u predict.py \
  --save_preds True \
  --save_vars 'tos' \
  --ckpt_list $ckpt_dir/seed1.bin \
  --config_path_list ./model_config.json \
  --dist_port $dist_port \
  --data_dir $data_dir \
  --input_var_list $input_var_list \
  --input_steps $input_steps \
  --predict_steps $predict_steps \
  --atmo_var_list tauu tauv \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --seed 1 \
  --log_level info \
  --dataloader_num_workers 8 \
  --per_device_eval_batch_size $batch_size
