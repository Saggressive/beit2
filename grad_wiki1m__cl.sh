device=0
alpha=1e-5
warmup_ratio=0.0
for LR in 1e-5 3e-5 5e-5 8e-5 1e-4 1.5e-4 2e-4
do
    sh run_condenser_wiki1m_cl.sh ${device} ${LR} ${alpha} ${warmup_ratio}
    device=`expr ${device} + 1`
done