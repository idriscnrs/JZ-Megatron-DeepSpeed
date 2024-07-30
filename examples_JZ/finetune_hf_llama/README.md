## Example of Finetuning LLAMA-7B from Hugging Face Weights

### Dataset
You can access the dataset from [here](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json).

### Pre-trained Weights
The pre-trained weights can be found at [Hugging Face - LLAMA-7B](https://huggingface.co/huggyllama/llama-7b).

### Installation on Jean Zay
```bash
git clone https://github.com/idriscnrs/JZ-Megatron-DeepSpeed.git
cd JZ-Megatron-DeepSpeed
module load pytorch-gpu/py3/2.3.1
pip install -e .
```

### Usage:

#### 1. Converting Hugging Face Model Weights to Megatron-Deepspeed Model
```bash
bash examples_deepspeed/finetune_hf_llama/finetune_llama.sh convert
```
This command writes the Hugging Face model weights into the Megatron-Deepspeed model and saves it. You can adjust the parallel configuration in the script.

#### 2. Fine-tuning Process
Use the following slurms script to fine-tune the model.

```bash
#! /bin/bash

#SBATCH --job-name=megadeep
#SBATCH --output=./h100.out
#SBATCH --error=./h100.out
#SBATCH -A sos@h100
#SBATCH -C h100
#SBATCH --time=10:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --exclusive


module purge
cd ${SLURM_SUBMIT_DIR}

module load pytorch-gpu/py3/2.3.1

set -x
pip list

DS_CONFIG=./examples_deepspeed/finetune_hf_llama/ds_config.json
DATASET_PATH=./alpaca_data.json
# dataset link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json

HF_LLAMA_PATH=/lustre/fsn1/projects/idris/sos/ssos022/models/Llama-2-7b-hf/
# weights link: https://huggingface.co/huggyllama/llama-7b

MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=256
TP=2
PP=2
# require to align with weight dimensions
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
NUM_LAYERS=32
NUM_HEADS=32
SEQ_LENGTH=512
######################################

MEGA_DS_LLAMA_PATH=./"llama-7b-mega-ds-T${TP}P${PP}"

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


covert_args="deepspeed tools/hf2megads_weight_converter.py \
--hf-ckpt-num-shards 2 \
--origin-hf-ckpt-dir $HF_LLAMA_PATH \
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

if [ "$1" = "convert" ]; then
    task_args="$covert_args"
else
    task_args="$finetune_args"
fi

full_cmd="$task_args $comm_args"

eval "$full_cmd"

```



