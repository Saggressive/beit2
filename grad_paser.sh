device=0
for seed in 1
do
    for LR in 3e-5 1e-4 2e-4 3e-4
    do
        alpha=1
        bash run_paser.sh ${device} ${LR} ${alpha} ${seed}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done

    for LR in 3e-5 1e-4 2e-4 3e-4
    do
        alpha=1
        bash run_paser_mim.sh ${device} ${LR} ${alpha} ${seed}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done

done
    