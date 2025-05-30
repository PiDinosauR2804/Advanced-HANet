for i in MAVEN
do
    for j in 1 2 3 4 42
    do
        for k in 5 10
        do
            python main.py \
                --data_root data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
                --class_num 20 \
                --backbone bert-base-uncased \
                --lr 2e-4 \
                --decay 1e-4 \
                --batch_size 4 \
                --device cuda:0 \
                --single_label \
                --dweight_loss \
                --num_description 3 \
                --sub_max \
                --cl_temp 0.07 \
                --skip_first_cl ucl+tlcl \
                --epochs 15 \
                --task_ep_time 2 \
                --uniform_ep 1 \
                --eval_freq 1 \
                --skip_eval_ep 0 \
                --patience 15 \
                --lora_rank 128 \
                --lora_alpha 128 \
                --lora_dropout 0.1 \
                --use_mole \
                --freeze_encoder_layers 0 \
                --mole_num_experts 8 \
                --mole_top_k 2 \
                --mole_level token \
                --mole_num_general_expert 1 \
                --general_expert_weight 1 \
                --scheduler lambda \
                --sheduler_type batch \
                --gammalr 0.95 \
                --step_size 1 \
                --stage_lr_ratio 0.7 \
                --warmup_ep 4 \
                --warmup_ratio 0.3 \
                --min_lr_ratio 0.1 \
                --eval_batch_size 256 \
                --gamma_router 1.01 \
                --balance_ratio 0.5 \
                --gate sigmoid \
                --ratio_loss_ucl 0.2 \
                --ratio_loss_tlcl 0.1 \
                --ratio_loss_aug 2 \
                --ratio_loss_fd 0.1 \
                --ratio_loss_pd 0.1 \
                --ratio_loss_rd 5 \
                --ratio_loss_lgacl 0.01 \
                --ratio_loss_des_cl 0.01 \
                --use_weight_ce \
                --alpha_ce 1.1 \
                --decrease_0_gpt_augmention \
                --aug_repeat_times 5 \
                --rep_aug mean \
                --cl_aug shuffle \
                --ucl \
                --distill pd rd\
                --use_description \
                --gpt_augmention \
                --joint_da_loss none \
                --project_name mole_maven_4_$k \
                --wandb
        done
    done
done
