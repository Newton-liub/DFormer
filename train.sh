GPUS=2
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CONFIG=${1:-local_configs.NYUDepthv2.DFormerPP_B}

export CUDA_VISIBLE_DEVICES="0,1"
export TORCHDYNAMO_VERBOSE=1

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config=$CONFIG --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --mst \
    --compile_mode="default" \
    --no-amp \
    --val_amp \
    --pad_SUNRGBD \
    --no-use_seed

# Usage:
#   bash train.sh local_configs.NYUDepthv2.DFormerPP_B
#   bash train.sh local_configs.NYUDepthv2.DFormerPP_S
#   bash train.sh local_configs.NYUDepthv2.DFormerPP_T
#   bash train.sh local_configs.NYUDepthv2.DFormer_Base
#   bash train.sh local_configs.NYUDepthv2.DFormerv2_B
#   bash train.sh local_configs.SUNRGBD.DFormerPP_B
