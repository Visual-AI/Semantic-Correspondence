CUDA_VISIBLE_DEVICES=0,1 accelerate  launch --mixed_precision=fp16 --main_process_port 8408 train.py \
  --dataset spair \
  --method dino \
  --pre_extract \
  --train_sample 2 \
  --val_sample 2 \
  --resolution 224 \
  --temperature 0.03 \
  --epochs 1 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --init_lr 0.0001 \
  --dino_lr 0.0001 \
  --scheduler "constant" \
  --save_thre 0.6 \
  --eval_interval 0.25 \
  --ckpt_dir ./0_spair/ \
  # --resume_dir ./0_ap10k/1207_0002_dino_True_ap10k_840_0.0001 \