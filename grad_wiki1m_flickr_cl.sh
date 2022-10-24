device=0
alpha=1
LR=3e-5
for seed in 1 2 3 4 5
do
    bash run_condenser_wiki1m_flickr_cl.sh ${device} ${alpha} ${LR} ${seed}
    device=`expr ${device} + 1`
done
