source activate zhangchenlong

for i in 3 4 5 
do
    for j in 3 4 5 6
    do
        python main.py \
            --data_root data/data_ids_enhence \
            --dataset ACE \
            --perm_id 0 \
            --seed 42 \
            --shot_num 5 \
            --class_num 10 \
            --backbone bert-base-uncased \
            --lr 5e-5 \
            --decay 1e-4 \
            --batch_size 4 \
            --device cuda:0 \
            --log \
            --log_dir ./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/ \
            --log_name ashuffle_lnone_r1 \
            --wandb \
            --project_name quangnm4_find_ace_5shot_4ep \
            --dweight_loss \
            --rep_aug mean \
            --distill mul \
            --single_label \
            --cl_aug shuffle \
            --aug_repeat_times 10 \
            --joint_da_loss ce \
            --sub_max \
            --cl_temp 0.07 \
            --ucl \
            --skip_first_cl ucl+tlcl \
            --use_description \
            --num_description 3 \
            --ratio_loss_des_cl 0.1 \
            --epochs $i \
            --task_ep_time $j \
            --uniform_ep 1 \
            --eval_freq 1 \
            --skip_eval_ep 0 \
            --patience 5 \
            --lora_rank 128 \
            --lora_alpha 128 \
            --lora_dropout 0.1 \
            --use_mole \
            --mole_num_experts 4 \
            --mole_top_k 1 \
            --mole_num_general_expert 1 \
            --entropy_weight 0.1 \
            --load_balance_weight 1 \
            --general_expert_weight 0.5 \
            --step_size 5 \
            --gammalr 0.95 \
            --eval_batch_size 256 \
            --gamma_router 1.01 \
            --balance_ratio 0.5 \
            --gate sigmoid \
            --ratio_loss_lgacl 0.1 \
            --gpt_augmention \
            --decrease_0_gpt_augmention \
            --ratio_loss_gpt 0.1
    done
done
