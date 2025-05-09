# kích hoạt môi trường
source activate zhangchenlong

# lặp qua 2 dataset ACE và MAVEN
for i in ACE MAVEN; do
    # gán class-num theo dataset
    if [ "$i" = "ACE" ]; then
        m=10
    else
        m=20
    fi
    # lặp qua shot-num = 5 và 10
    for j in 5 10
        for k in gpt_augmention decrease_0_gpt_augmention gpt_augmention+decrease_0_gpt_augmention
            for l in 0.7 0.5 0.3; do
                GPT=""
                DECREASE=""

                if [[ "$k" == "gpt_augmention" || "$k" == "gpt_augmention+decrease_0_gpt_augmention" ]]; then
                    GPT="--gpt_augmention"
                fi
                if [[ "$k" == "decrease_0_gpt_augmention" || "$k" == "gpt_augmention+decrease_0_gpt_augmention" ]]; then
                    DECREASE="--decrease_0_gpt_augmention"
                fi
                python classifier/train.py \
                    --data-root ./data/data_ids_enhence \
                    --dataset "$i" \
                    --backbone bert-base-uncased \
                    --lr 2e-5 \
                    --decay 1e-4 \
                    --no-freeze-bert \
                    --shot-num "$j" \
                    --batch-size 16 \
                    --device cuda:0 \
                    --log \
                    --log-dir ./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/ \
                    --log-name "${i}_${j}shot" \
                    --dweight_loss \
                    --rep-aug mean \
                    --distill pd \
                    --epochs 30 \
                    --class-num "$m" \
                    --single-label \
                    --cl-aug shuffle \
                    --aug-repeat-times 5 \
                    --joint-da-loss ce \
                    --sub-max \
                    --cl_temp 0.07 \
                    --ucl \
                    --skip-first-cl ucl+tlcl \
                    --use-description \
                    --num_description 3 \
                    --ratio_loss_des_cl 0.1 \
                    $GPT \
                    $DECREASE \
                    --use_weight_ce \
                    --alpha_ce $l
            done
        done
    done
done
