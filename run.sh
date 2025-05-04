!python main.py \
    --data_root ./data/data_ids \
    --dataset MAVEN \
    --backbone bert-base-uncased \
    --lr 2e_5 \
    --decay 1e_4 \
    --no_freeze_bert \
    --shot_num 5 \
    --batch_size 16 \
    --device cuda:0 \
    --log \
    --log_dir ./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/ \
    --log_name ashuffle_lnone_r1 \
    --dweight_loss \
    --rep_aug mean \
    --distill pd \
    --epoch 30 \
    --class_num 20 \
    --single_label \
    --cl_aug shuffle \
    --aug_repeat_times 5 \
    --joint_da_loss ce \
    --sub_max \
    --cl_temp 0.07 \
    --ucl \
    --skip_first_cl ucl+tlcl \
    --use_description \
    --num_description 3 \
    --ratio_loss_des_cl 0.1  
    # --task_ep_time 2 \
    # --early_stop \
    # --skip_eval_ep 30 \
    # --eval_freq 5 \
    # --patience 2 \
    # --freeze_embedding_layer \
    # --freeze_encoder_layers 6 \
    # --classifier_layer 1 \
    # --hidden_dim 128 \
    # --dropout 0.5 \
    # --lora_rank 128 \
    # --lora_alpha 32 \
    # --use_mole \
    # --mole_num_experts 4 \
    # --mole_top_k 2 \
    # --uniform_ep 20 \
    # --use_general_expert \
    # --entropy_weight 0.1 \
    # --load_balance_weight 1 \
    # --general_expert_weight 0.1 \
    # --step_size 30 \
    # --gammalr 0.2