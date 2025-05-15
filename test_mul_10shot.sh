source activate zhangchenlong

for i in 2 3 4 
do
    for j in 3 4 5 6
    do
        for k in 0.1 0.3 0.5 0.7 0.9
        do
            for l in 0.1
            do
                for n in mul
                do
                    if [ "$j" = "ACE" ]; then
                        t=10
                    else
                        t=20
                    fi

                    python main.py \
                        --data_root ./data/data_ids_enhence \
                        --dataset ACE \
                        --perm_id 0 \
                        --seed 42 \
                        --shot_num 10 \
                        --class_num $t \
                        --backbone bert-base-uncased \
                        --lr 2e-5 \
                        --decay 1e-4 \
                        --no_freeze_bert \
                        --batch_size 4 \
                        --device cuda:0 \
                        --log \
                        --log_dir ./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/ \
                        --log_name ashuffle_lnone_r1 \
                        --wandb \
                        --project_name ACE_10_shot_16_5_mul \
                        --dweight_loss \
                        --rep_aug mean \
                        --distill $n \
                        --single_label \
                        --cl_aug shuffle \
                        --aug_repeat_times 9 \
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
                        --eval_freq 2 \
                        --skip_eval_ep 0 \
                        --patience 4 \
                        --lora_rank 64 \
                        --lora_alpha 64 \
                        --lora_dropout 0.3 \
                        --use_lora \
                        --use_mole \
                        --mole_num_experts 4 \
                        --mole_top_k 2 \
                        --use_general_expert \
                        --entropy_weight 0.1 \
                        --load_balance_weight 1 \
                        --general_expert_weight 0.2 \
                        --step_size 1 \
                        --gammalr 0.97 \
                        --eval_batch_size 256 \
                        --eval_ratio 0.25 \
                        --gpt_augmention \
                        --decrease_0_gpt_augmention \
                        --ratio_loss_gpt $l \
                        --use_weight_ce \
                        --alpha_ce $k
                done
            done
        done
    done
done
