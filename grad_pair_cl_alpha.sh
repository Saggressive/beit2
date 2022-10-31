device=0

for seed in 1 2 3 4 5
do
    for alpha in 1 0.1 0.01 1e-3 1e-4 1e-6
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
    