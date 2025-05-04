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
    for j in 5 10; do
        python classifier/train.py \
            --data-root ./data/data_ids_aug \
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
            --ratio_loss_des_cl 0.1     
    done
done
