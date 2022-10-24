device=0
for seed in 1 2 3 4 5
do
    for alpha in 1e-6 1e-7
    do  
        LR=3e-5
        bash run_condenser_beit_mim.sh ${device} ${alpha} ${LR} ${seed}
        device=`expr ${device} + 1`
    done

done
    