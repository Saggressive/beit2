device=0

for seed in 1 2 3 4 5
do
    for mask in 20 40 59 75 98 112 137 157 176
    do  
        alpha=1e-5
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
    