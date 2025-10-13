# run remote

CONFIG="rae_imagenet"
BATCH_SIZE=1024
WORKDIR="RAE-XL"
BUCKET="$GCS_BUCKET"

: "${WANDB_API_KEY:?Set WANDB_API_KEY before launching the training job.}"
: "${GCS_BUCKET:?Set GCS_BUCKET before launching the training job.}"

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592

# only RAE eval is supported at the moment 
WANDB_API_KEY="$WANDB_API_KEY" python main.py \
    --workdir=$WORKDIR \
    --bucket=$BUCKET \
    --config=configs/$CONFIG.py:imagenet_raw_256-XL_1 \
    --config.data.batch_size=$BATCH_SIZE \
    --config.standalone_eval=True \
    --config.project_name='diffuse_nnx' \
    --config.eval.on_load=False \
    --config.visualize.on=True \
    --config.exp_name='RAE-XL' \
