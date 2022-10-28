beta1=$1
beta2=$2
device=0
for seed in 1 2 3 4 5
do
    for LR in ${beta1} ${beta2}
    do
        for beta in 1e-3 5e-3 0.01 0.03
        do  
            temp_v=0.03
            bash run_condenser_pair_cl.sh ${device} ${beta} ${temp_v} ${LR} ${seed}
            device=`expr ${device} + 1`
            if [ ${device} == 8 ]
            then
                sleep 20
                device=0
            fi
        done

        for temp_v in 0.01 0.05 0.07 0.1 0.2
        do  
            beta=0.01
            bash run_condenser_pair_cl.sh ${device} ${beta} ${temp_v} ${LR} ${seed}
            device=`expr ${device} + 1`
            if [ ${device} == 8 ]
            then
                sleep 20
                device=0
            fi
        done
    done

done
    