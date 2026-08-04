# Train on the unified reid dataset (single GPU).
# usage: sh run_reid.sh <arch: small|base> <vram: 8gb|16gb|96gb> [device] [pretrain]
arch=${1:-small}
vram=${2:-8gb}
device=${3:-0}
pretrain=${4:-}

config=configs/reid/vit_${arch}_${vram}.yml
if [ ! -f "$config" ]; then
    echo "no config for arch=${arch} vram=${vram} (${config})"
    exit 1
fi

extra=""
if [ -n "$pretrain" ]; then
    extra="MODEL.PRETRAIN_PATH ${pretrain}"
fi

python train.py --config_file ${config} \
    MODEL.DEVICE_ID "('${device}')" \
    OUTPUT_DIR logs/reid_vit_${arch}_${vram} \
    ${extra}
