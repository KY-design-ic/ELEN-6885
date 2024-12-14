def load_toxic_dataset(dataset_name: str, toxicity_threshold=0.3, cache_dir=None):
    """
    Load a dataset from huggingface by using `load_dataset`


    Args:
        dataset_name (`str`): a dataset to be loaded from huggingface.
        toxicity_threshold (`float32`): threshold to determine if an input is toxic.
        cache_dir (`str`): path to cache pretrained LLM.

    Returns:
        a dataset
    """
    if dataset_name == "allenai/real-toxicity-prompts":
        ds = load_dataset(dataset_name, split="train")

        def select_toxic(sample):
            toxicity = sample["prompt"]["toxicity"]
            return toxicity is not None and toxicity > toxicity_threshold
    elif dataset_name == "jigsaw_unintended_bias":
        dataset_name = "./dataset/test_public_expanded.csv"
        ds = load_dataset("csv", data_files=dataset_name)['train']
        def select_toxic(sample):
            toxicity = sample["toxicity"]
            return toxicity is not None and toxicity > toxicity_threshold
    else:
        raise ValueError("No such dataset used in the experiment.")

    ds = ds.filter(select_toxic, batched=False)
    return ds

def get_tokenized(tokenizer, dataset, dataset_name, min_text_length, max_text_length):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        tokenizer : a huggingface tokenizer for text tokenization.
        dataset (`dataset.Dataset`): A huggingface dataset to be loaded.
        dataset_name (`str`): dataset name which helps to determine the way to tokenize.
        min_text_length (`int`): minimal length of input.
        max_text_length (`int`): maximal length of input.

    Returns:
        dataloader (`torch.utils.data.DataLoader`): a dataloader for the dataset used in the training loop.
    """

    # sample by input length
    input_size = LengthSampler(min_text_length, max_text_length)

    def tokenize_real_toxicity_prompts(sample):
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]

        sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    def tokenize_jigsaw_unintended_bias(sample):
        prompt = sample["comment_text"]

        sample["input_ids"] = tokenizer.encode(prompt)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    if dataset_name == "allenai/real-toxicity-prompts":
        dataset = dataset.map(tokenize_real_toxicity_prompts, batched=False)
    else:
        dataset = dataset.map(tokenize_jigsaw_unintended_bias, batched=False)

    dataset.set_format(type="torch")

    return dataset

def load_pretrained_model_tokenizer(model_name_or_path, device="cpu", cache_dir=None):
    """Loads a trained model from the given model name or path."""
    tokenizer = get_tokenizer(model_name_or_path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,  # change to location you want to store the pretrained-model
        pad_token_id=tokenizer.eos_token_id,
        # torch_dtype=torch.bfloat16 ## use torch.bfloat16 to save memory
    )
    model = model.to(device)
    return model, tokenizer
def get_tokenizer(model_name_or_path, cache_dir):
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# experiment config
min_text_length = 20
max_text_length = 40
cache_dir = "./cache"
model_name_or_path = "EleutherAI/gpt-neo-125m"
device = "cpu"

# load model and tokenizer
model, tokenizer = load_pretrained_model_tokenizer(model_name_or_path, device=device, cache_dir=cache_dir)

# load dataloader for dataset: "allenai/real-toxicity-prompts" or "jigsaw_unintended_bias"
# dataset_name = "allenai/real-toxicity-prompts" # either "jigsaw_unintended_bias" or "allenai/real-toxicity-prompts"
dataset_name = "jigsaw_unintended_bias"
toxicity_threshold = 0.3
test_ratio = 0.2
dataset = load_toxic_dataset(dataset_name=dataset_name, toxicity_threshold=toxicity_threshold, cache_dir=cache_dir)
dataset = dataset.train_test_split(test_size=test_ratio, shuffle=False)
train_dataset = get_tokenized(tokenizer, dataset['train'], dataset_name, min_text_length, max_text_length)
test_dataset = dataset['test']

train_dataset

test_dataset

## 1. create reference model using traditional fine tuning
model = AutoModelForCausalLMWithValueHead.from_pretrained("EleutherAI/gpt-neo-125m")

num_layers = len(model.pretrained_model.transformer.h)
print(f"number of layers in total: {num_layers}")
num_shared_layers = num_layers - 8 ## tuning on last two layers and freeze other layers
reference_model = create_reference_model(model, num_shared_layers=num_shared_layers)

#from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=128,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Assuming 'model' is your base AutoModelForCausalLMWithValueHead
#model = get_peft_model(model, lora_config)
# Now you have two models:
# 1. model_original: The original model for traditional fine-tuning.
# 2. model_peft: The new model for PEFT with the LoRA configuration.


import trl
import transformers
import peft

print("TRL version:", trl.__version__)
print("Transformers version:", transformers.__version__)
print("PEFT version:", peft.__version__)

## 2. create reference model using peft
from peft import LoraConfig

lora_config = LoraConfig(
    r=128,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained("EleutherAI/gpt-neo-125m", peft_config=lora_config)