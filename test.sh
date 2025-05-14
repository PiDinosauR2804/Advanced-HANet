source activate zhangchenlong

for i in ucl tlcl ucl+tlcl none
do
    for j in ACE MAVEN
    do
        for l in ce none
        do
            for m in 5
            do
                for n in 0.1 0.3 0.5 0.7
                do
                    if [ "$j" = "ACE" ]; then
                        t=10
                    else
                        t=20
                    fi
                    # Tạo flags conditionally
                    UCL_FLAG=""
                    TLCL_FLAG=""

                    if [[ "$i" == "ucl" || "$i" == "ucl+tlcl" ]]; then
                        UCL_FLAG="--ucl"
                    fi
                    if [[ "$i" == "tlcl" || "$i" == "ucl+tlcl" ]]; then
                        TLCL_FLAG="--tlcl"
                    fi

                    python classifier/train.py \
                        --data-root ./data/data_ids \
                        --dataset $j \
                        --backbone bert-base-uncased \
                        --lr 2e-5 \
                        --decay 1e-4 \
                        --no-freeze-bert \
                        --shot-num $m \
                        --batch-size 4 \
                        --device cuda:0 \
                        --log \
                        --log-dir ./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/ \
                        --log-name ashuffle_lnone_r1 \
                        --dweight_loss \
                        --rep-aug mean \
                        --distill mul \
                        --epoch 30 \
                        --class-num $t \
                        --single-label \
                        --cl-aug shuffle \
                        --aug-repeat-times 5 \
                        --joint-da-loss $l \
                        --sub-max \
                        --cl_temp 0.07 \
                        $UCL_FLAG \
                        $TLCL_FLAG \
                        --skip-first-cl ucl+tlcl \
                        --use-description \
                        --num_description 3 \
                        --ratio_loss_des_cl $n     
                done
            done
        done
    done
done
