device=0
for seed in 1 2 3 4 5
do
    for LR in 5e-6 8e-6 1e-5 2e-5
    do
        r=0.1
        alpha=1e-5
        mask=118
        e=2
        bash run_warmup.sh ${device} ${LR} ${alpha} ${mask} ${r} ${seed} ${e}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done

    for LR in 5e-6 8e-6 1e-5 2e-5
    do
        r=0.2
        alpha=1e-5
        mask=118
        e=2
        bash run_warmup.sh ${device} ${LR} ${alpha} ${mask} ${r} ${seed} ${e}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done

done
    