#!/bin/bash

# Wait for prelog to finish
echo "Wait 3s..."
sleep 3
echo "done"

PRELOAD="singularity exec --bind $SCRATCH/hylo:$HOME,$SCRATCH/kaggle_3m --nv $SCRATCH/hylo-box "

# Args of training script
CMD="main-dist-segmentation.py --data-dir $SCRATCH/kaggle_3m/ --freq 200  --dataset brain-segmentation --batch-size 16 --model unet --epochs 50 --warmup-epochs 10 --milestone 100 --lr 0.0008 --lr-decay 1 --damping 0.03 --target-damping 0.03 --weight-decay 0.0001 --nproc-per-node 4 --adaptive 10 50 100 $@"

# Distributed system configuration
if [[ -z "${NODEFILE}" ]]; then
    if [[ -n "${SLURM_NODELIST}" ]]; then
        NODEFILE=/tmp/slurm_nodelist
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
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=4"

FULL_CMD="$PRELOAD $LAUNCHER $CMD"
echo "Training command: $FULL_CMD"

# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do
    FULL_CMD+=" --node-idx $RANK --url tcp://$MAIN_RANK:1223"
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
