#!/bin/bash

# Wait for prolog to finish
echo "Wait 3s..."
sleep 3
echo "done"

PRELOAD="singularity exec --bind $SCRATCH/hylo:$HOME,$SCRATCH/cifar100 --nv artifact_init.sif "

# Args of training script
CMD="main-dist-classification.py --freq 100 --dataset fashion-mnist --data-dir $SCRATCH/fashion-mnist --batch-size 128 --model 3c1f --epochs 10 --warmup-epochs 0 --milestone 30 45 --lr 0.003 --lr-decay 0.1 --damping 0.3 --target-damping 0.3 --damping-decay-epochs 10 --weight-decay 0.001 --checkpoint-freq 10 --adaptive 0 10 10 --compression-ratio 1 --log-dir convnet-fashion-mnist-end-to-end $@"

# Distributed system configuration
if [[ -z "${NODEFILE}" ]]; then
    if [[ -n "${SLURM_NODELIST}" ]]; then
        NODEFILE=/tmp/cifar100_slurm_nodelist
        scontrol show hostnames $SLURM_NODELIST > $NODEFILE
    elif [[ -n "${COBALT_NODEFILE}" ]]; then
        NODEFILE=$COBALT_NODEFILE
    fi
fi
if [[ -z "${NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi

# Torch distributed launcher
LAUNCHER="python -m torch.distributed.launch "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=1" # single GPU experiment, make only 1 GPU in the node visible to the program

FULL_CMD="$PRELOAD $LAUNCHER $CMD"
echo "Training command: $FULL_CMD"

# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do
    FULL_CMD+=" --url tcp://$MAIN_RANK:1234"
    echo "node idx cmd: $FULL_CMD"
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK+1))
done

wait
