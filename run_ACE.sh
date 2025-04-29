source activate zhangchenlong
for i in ucl tlcl ucl+tlcl none
do
    for j in 5
    do
        for k in shuffle RTR dropout none
        do
            for l in none
            do
                for m in 10
                do
                    for n in fd pd mul none
                    do
                        python classifier/train.py \
                            --data-root ./data/data_ids \
                            --dataset ACE \
                            --backbone bert-base-uncased \
                            --lr 2e-5 \
                            --decay 1e-4 \
                            --no-freeze-bert \
                            --shot-num $j \
                            --batch-size 16 \
                            --device cuda:0 \
                            --log \
                            --log-dir ./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/ \
                            --log-name a${k}_l${l}_r${i} \
                            --dweight_loss \
                            --rep-aug mean \
                            --distill $n \
                            --epoch 30 \
                            --class-num $m \
                            --single-label \
                            --cl-aug $k \
                            --aug-repeat-times 5 \
                            --joint-da-loss $l \
                            --sub-max \
                            --cl_temp 0.07 \
                            --tlcl \
                            --ucl \
                            --skip-first-cl $i
                    done
                done
            done
        done
    done
done
