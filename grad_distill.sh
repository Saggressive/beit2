device=0
for LR in 5e-5 1e-4 1e-5 3e-5
do
    alpha=1
    bash run_distill_mse.sh ${device} ${LR} ${alpha}
    device=`expr ${device} + 1`
    if [ ${device} == 8 ]
    then
        sleep 20
        device=0
    fi
done

for alpha in 0.1 0.25 0.5 0.75
do
    LR=5e-5
    bash run_distill_mse.sh ${device} ${LR} ${alpha}
    device=`expr ${device} + 1`
    if [ ${device} == 8 ]
    then
        sleep 20
        device=0
    fi
done