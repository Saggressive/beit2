device=0
for seed in 1 2 3 4 5
do
    for LR in 8e-6 1e-5 2e-5 3e-5
    do
        alpha=1e-5
        mask=75
        bash run_coco_simcse.sh ${device} ${LR} ${seed}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done

done
    