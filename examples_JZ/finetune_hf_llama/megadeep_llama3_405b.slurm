#! /bin/bash

#SBATCH --job-name=megadeep_llama3_405b
#SBATCH --output=./megadeep_llama3_405b.out
#SBATCH --error=./megadeep_llama3_405b.out
#SBATCH -A sos@h100
#SBATCH -C h100
#SBATCH --time=10:30:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --exclusive


module purge
cd ${SLURM_SUBMIT_DIR}

module load pytorch-gpu/py3/2.4.0

set -x
pip list

DS_CONFIG=./examples_deepspeed/finetune_hf_llama/ds_config.json
DATASET_PATH=./alpaca_data.json
# dataset link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json

HF_LLAMA_PATH=/lustre/fsn1/projects/idris/sos/ssos022/models/Meta-Llama-3-70B-Instruct
# weights link: https://huggingface.co/huggyllama/llama-7b

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64
TP=8
PP=64
# require to align with weight dimensions
HIDDEN_SIZE=16384
FFN_HIDDEN_SIZE=53248
NUM_LAYERS=126
NUM_HEADS=128
NUM_KEY_VALUE_HEADS=16
SEQ_LENGTH=2048
HF_CKPT_NUM_SHARDS=191
######################################

MEGA_DS_LLAMA_PATH=./"llama-405b-mega-ds-T${TP}P${PP}"

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 100,
  "zero_optimization": {
    "stage": 0
  },
  "bf16": {
    "enabled": true
  }
}
EOT


covert_args="srun python tools/hf2megads_weight_converter.py \
--hf-ckpt-num-shards $HF_CKPT_NUM_SHARDS \
--origin-hf-ckpt-dir $HF_LLAMA_PATH \
--load-safetensors \
--save $MEGA_DS_LLAMA_PATH"

finetune_args="srun python finetune_llama.py \
--load $MEGA_DS_LLAMA_PATH"

comm_args="--tensor-model-parallel-size $TP \
--pipeline-model-parallel-size $PP \
--lr-warmup-iters 2000 \
--weight-decay 0.1 \
--clip-grad 1 \
--num-layers $NUM_LAYERS \
--hidden-size $HIDDEN_SIZE \
--num-attention-heads $NUM_HEADS \
--num-key-value-heads $NUM_KEY_VALUE_HEADS \
--ffn-hidden-size $FFN_HIDDEN_SIZE \
--attention-dropout 0 \
--hidden-dropout 0 \
--no-query-key-layer-scaling \
--disable-bias-linear \
--normalization rmsnorm \
--use-rotary-position-embeddings \
--untie-embeddings-and-output-weights \
--swiglu \
--seq-length $SEQ_LENGTH \
--max-position-embeddings $SEQ_LENGTH \
--micro-batch-size $MICRO_BATCH_SIZE \
--global-batch-size $GLOBAL_BATCH_SIZE \
--train-iters 3500 \
--lr 2e-5 \
--tensorboard-dir tensorboard_output \
--lr-decay-iters 320000 \
--lr-decay-style cosine \
--log-interval 1 \
--eval-iters 100 \
--eval-interval 100 \
--data-path $DATASET_PATH \
--save-interval 1500 \
--split 100,0,0 \
--bf16 \
--zero-stage 0 \
--tokenizer-type HFTokenizer \
--tokenizer-model $HF_LLAMA_PATH \
--deepspeed_config ./examples_deepspeed/finetune_hf_llama/ds_config.json \
--deepspeed \
--distributed-backend nccl \
--num-workers 0 \
--no-masked-softmax-fusion \
--no-bias-gelu-fusion \
--no-bias-dropout-fusion \
--no-gradient-accumulation-fusion \
--repeated-dataloader \
--save ./out_save \
--finetune"

full_cmd="$finetune_args $comm_args"

eval "$full_cmd"


