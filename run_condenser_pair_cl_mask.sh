export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
# node_rank=0
device=$1
export CUDA_VISIBLE_DEVICES=${device}
beta=$2
temp_v=$3
LR=$4
mask=$5
name=pair_cl_mask_temp_v${temp_v}_beta${beta}_LR${LR}_mask${mask}
# all_dir=./save/condenser/${name}_${temp_v}_${a0}_${a1}_${a2}_${a3}_epoch50
all_dir=save/pair_cl_mask/${name}
log_dir=save/pair_cl_mask/tensorboard_log/${name}
mkdir -p ${all_dir}
mkdir -p ${log_dir}
nohup python run_mib_pretraining.py \
    --accum_iter 1 \
    --data_set image_folder \
    --paired_data_path ir_data/flickr_random_captions.json \
    --text_data_path ir_data/wiki1m.txt \
    --output_dir ${all_dir} \
    --log_dir ${log_dir} \
    --model beit_base_patch16_224_8k_vocab_cls_pt \
    --shared_lm_head True \
    --early_layers 9 \
    --head_layers 2 \
    --num_mask_patches ${mask} \
    --second_input_size 224 \
    --second_interpolation bicubic \
    --min_crop_scale 0.2 \
    --tokenizer_model vqkd_encoder_base_decoder_3x768x12_clip \
    --tokenizer_weight ./pretrained_model/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth \
    --resume /nlp_group/wuxing/suzhenpeng/beit2/pretrained_model/beitv2_base_patch16_224_pt1k.pth\
    --batch_size 64 \
    --lr ${LR} \
    --clip_grad 1.0 \
    --drop_path 0.1 \
    --layer_scale_init_value 0.1 \
    --imagenet_default_mean_and_std \
    --opt_betas 0.9 0.999 \
    --opt_eps 1e-8  \
    --weight_decay 0.00 \
    --epochs 1 \
    --save_ckpt_freq 20 \
    --init_condenser \
    --warmup_ratio 0.0 \
    --model_name_or_path pretrained_model/condenser \
    --use_text_cl \
    --use_pair_cl \
    --temp 0.05 \
    --temp_v ${temp_v} \
    --alpha 1 \
    --beta ${beta} \
    --max_seq_length 32 \
    --train_mode all \
    --seed 0 \
    --a0 1 \
    --a1 1 \
    --a2 1 \
    --a3 1 \
    >${all_dir}/${name}_${node_rank}.log 2>&1 &
if [ ${device} == 7 ]
then
    wait
fi