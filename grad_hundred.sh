device=0
for seed in 1 2 3 4 5
do
    for LR in 8e-6 1e-5 2e-5 3e-5
    do
        alpha=1e-5
        bash run_hundred.sh ${device} ${LR} ${alpha} ${seed}
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
        bash run_hundred.sh ${device} ${LR} ${alpha} ${seed}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done

done
    