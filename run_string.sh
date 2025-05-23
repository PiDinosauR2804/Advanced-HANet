source activate zhangchenlong

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
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
                --project_name output_best_setting_MAVEN \
                --save_output output_best_setting_MAVEN \
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
                --epochs 3 \
                --task_ep_time 6 \
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
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done

# Table 2

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
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
                --project_name table_2_no_description \
                --save_output table_2_no_description \
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
                --epochs 3 \
                --task_ep_time 6 \
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
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
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
                --project_name table_2_no_augmentation \
                --save_output table_2_no_augmentation \
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
                --epochs 3 \
                --task_ep_time 6 \
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
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
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
                --project_name table_2_no_both \
                --save_output table_2_no_both \
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
                --epochs 3 \
                --task_ep_time 6 \
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
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done

# Table 3

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
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
                --project_name table3_1expert \
                --save_output table3_1expert \
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
                --epochs 3 \
                --task_ep_time 6 \
                --uniform_ep 1 \
                --eval_freq 2 \
                --skip_eval_ep 0 \
                --patience 4 \
                --lora_rank 64 \
                --lora_alpha 64 \
                --lora_dropout 0.3 \
                --use_lora \
                --use_mole \
                --mole_num_experts 1 \
                --mole_top_k 1 \
                --use_general_expert \
                --entropy_weight 0.1 \
                --load_balance_weight 1 \
                --general_expert_weight 0.2 \
                --step_size 1 \
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
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
                --project_name table3_8expert \
                --save_output table3_8expert \
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
                --epochs 3 \
                --task_ep_time 6 \
                --uniform_ep 1 \
                --eval_freq 2 \
                --skip_eval_ep 0 \
                --patience 4 \
                --lora_rank 64 \
                --lora_alpha 64 \
                --lora_dropout 0.3 \
                --use_lora \
                --use_mole \
                --mole_num_experts 8 \
                --mole_top_k 4 \
                --use_general_expert \
                --entropy_weight 0.1 \
                --load_balance_weight 1 \
                --general_expert_weight 0.2 \
                --step_size 1 \
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
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
                --project_name table3_12expert \
                --save_output table3_12expert \
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
                --epochs 3 \
                --task_ep_time 6 \
                --uniform_ep 1 \
                --eval_freq 2 \
                --skip_eval_ep 0 \
                --patience 4 \
                --lora_rank 64 \
                --lora_alpha 64 \
                --lora_dropout 0.3 \
                --use_lora \
                --use_mole \
                --mole_num_experts 12 \
                --mole_top_k 4 \
                --use_general_expert \
                --entropy_weight 0.1 \
                --load_balance_weight 1 \
                --general_expert_weight 0.2 \
                --step_size 1 \
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done

# Table 4

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
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
                --project_name table4_3augment \
                --save_output table4_3augment \
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
                --epochs 3 \
                --task_ep_time 6 \
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
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1 \
                --num_augmention 3 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
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
                --project_name table4_5augment \
                --save_output table4_5augment \
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
                --epochs 3 \
                --task_ep_time 6 \
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
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1 \
                --num_augmention 5 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done

# Table 5

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
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
                --project_name table5_1des \
                --save_output table5_1des \
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
                --num_description 1 \
                --ratio_loss_des_cl 0.1 \
                --epochs 3 \
                --task_ep_time 6 \
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
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
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
                --project_name table5_5des \
                --save_output table5_5des \
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
                --num_description 5 \
                --ratio_loss_des_cl 0.1 \
                --epochs 3 \
                --task_ep_time 6 \
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
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done

# Table 6

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
                --class_num $t \
                --backbone bert-base-uncased \
                --lr 2e-5 \
                --decay 1e-4 \
                --batch_size 4 \
                --device cuda:0 \
                --log \
                --log_dir ./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/ \
                --log_name ashuffle_lnone_r1 \
                --wandb \
                --project_name table6_free_bert \
                --save_output table6_free_bert \
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
                --epochs 3 \
                --task_ep_time 6 \
                --uniform_ep 1 \
                --eval_freq 2 \
                --skip_eval_ep 0 \
                --patience 4 \
                --lora_rank 64 \
                --lora_alpha 64 \
                --lora_dropout 0.3 \
                --use_lora \
                --use_mole \
                --mole_num_experts 0 \
                --mole_top_k 0 \
                --use_general_expert \
                --entropy_weight 0.1 \
                --load_balance_weight 1 \
                --general_expert_weight 0.2 \
                --step_size 1 \
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done

# Table 4 remain

for i in ACE MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
                tt=0.3
            else
                t=20
                tt=0.3
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
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
                --project_name table4_7augment \
                --save_output table4_7augment \
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
                --epochs 3 \
                --task_ep_time 6 \
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
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1 \
                --num_augmention 7 \
                --use_weight_ce \
                --alpha_ce $tt
        done
    done
done
