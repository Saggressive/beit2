device=0
for seed in 1 2 3 4 5
do
    for LR in 1e-5 2e-5
    do
        alpha=1e-5
        mask=75
        bash run_head_norm.sh ${device} ${LR} ${alpha} ${mask} ${seed}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done

    for alpha in 1e-1 1e-2 1e-3 1e-4
    do  
        LR=8e-6
        mask=75
        bash run_head_norm.sh ${device} ${LR} ${alpha} ${mask} ${seed}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done

    for mask in 98 118
    do  
        LR=8e-6
        alpha=1e-5
        bash run_head_norm.sh ${device} ${LR} ${alpha} ${mask} ${seed}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done

done
    