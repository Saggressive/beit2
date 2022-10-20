node=$1
lr=3e-5
a0=1
a3=1
for a in 1
    do
        sh run_pretrain.sh ${node} ${lr} ${a0} ${a} ${a} ${a3}
        wait
    done