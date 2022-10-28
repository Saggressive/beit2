device=0
for mask in 75 98 112 137 157 176
do
    for alpha in 1e-3 1e-4 1e-5
    do  
        LR=3e-5
        bash run_condenser_beit_mim.sh ${device} ${alpha} ${LR} ${mask}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 30
            device=0
        fi
    done

    for LR in 2e-5 1e-5 4e-5
    do  
        alpha=1e-5
        bash run_condenser_beit_mim.sh ${device} ${alpha} ${LR} ${mask}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 30
            device=0
        fi
    done
done
    