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
sbatch examples_JZ/finetune_hf_llama/convert_weight_llama2_7b.slurm
```
This command writes the Hugging Face model weights into the Megatron-Deepspeed model and saves it. You can adjust the parallel configuration in the script.

#### 2. Fine-tuning Process
Use the following slurms script to fine-tune the model.

```bash
sbatch examples_JZ/finetune_hf_llama/megadeep_llama2_7b.slurm
```



