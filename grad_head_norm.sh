device=0
for seed in 1 2 3 4 5
do
    for LR in 6e-6 7e-6
    do
        alpha=1e-5
        mask=118
        decay=0
        bash run_head_norm.sh ${device} ${LR} ${alpha} ${mask} ${decay} ${seed}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done

    for decay in 0.01 0.03 0.05
    do
        alpha=1e-5
        mask=118
        LR=8e-6
        bash run_head_norm.sh ${device} ${LR} ${alpha} ${mask} ${decay} ${seed}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done

    for mask in 137 157 167
    do     
        decay=0
        LR=8e-6
        alpha=1e-5
        bash run_head_norm.sh ${device} ${LR} ${alpha} ${mask} ${decay} ${seed}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done

done
    