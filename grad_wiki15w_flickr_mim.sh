device=0
for seed in 1 2 3 4 5
do
    for alpha in 1e-2 1e-3 1e-4 1e-5
    do  
        LR=3e-5
        bash run_wiki15w_flickr_mim.sh ${device} ${alpha} ${LR} ${seed}
        device=`expr ${device} + 1`
    done

    for LR in 2e-5 1e-5 4e-5 5e-5
    do  
        alpha=1e-5
        bash run_wiki15w_flickr_mim.sh ${device} ${alpha} ${LR} ${seed}
        device=`expr ${device} + 1`
    done
    wait
    device=0
done
    