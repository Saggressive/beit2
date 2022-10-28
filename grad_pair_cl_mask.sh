device=0

for seed in 0 1 2 3 4 5
do
    for alpha in 0.02 0.05 0.1 0.2 0.005
    do  
        mask=75
        bash run_condenser_pair_cl_mimsearch.sh ${device} ${alpha} ${mask} ${seed}
        device=`expr ${device} + 1`
        if [ ${device} == 8 ]
        then
            sleep 20
            device=0
        fi
    done
    wait
done
    