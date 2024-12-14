from typing import List, Union, Dict, Any
from datasets import Dataset

def tokenize_data(
    tokenizer, input_data: Union[List[str], Dataset], config: Optional[Dict[str, Any]] = None):
    """Tokenize text data.

    Args:
        input_data: The text to be tokenized.
        tokenizer: a tokenizer to tokenize.
        config: parameters for setting up the tokenization. Defaults to None.

    Returns:
        tokenized data Dict[str, Tensor]: tokenized data with input_ids, attention_masks and labels.
    """
    if not isinstance(input_data, List):
        input_data = input_data["text"]

    encoded: Dict[str, torch.Tensor] = tokenizer(
        input_data,
        padding=True,
        # truncation=True,
        return_tensors="pt",
    )
    return encoded

def tokenize_on_dataset(tokenizer, dataset: Dataset, config: Optional[Dict[str, Any]] = None):
    """main function to perform tokenization over a dataset object

    Args:
        tokenizer (PreTrainedTokenizer): a tokenizer
        dataset (Dataset): a dataset object to be tokenized, the feature *text* will be tokenized.
        config (Dict, optional): parameters for setting up the tokenization. Defaults to None.

    Returns:
        tokenized data Dict[str, Tensor]: tokenized data with input_ids, attention_masks and labels.
    """
    tokenized_dataset = dataset.map(lambda x: tokenize_data(tokenizer, x, None), batched=True, num_proc=4)
    return tokenized_dataset

in_dist_ds

# in-distribution test - coming from the same dataset that we used to train our detoxified model
in_dist_ds = test_dataset
def get_text(x):
    if "real" in dataset_name:
        x['text'] = x['prompt']["text"][:context_length]
    else:
        x['text'] = x["comment_text"][:context_length]
    return x

in_dist_ds = in_dist_ds.map(get_text, batched=False)
in_dist_ds = tokenize_on_dataset(tokenizer, in_dist_ds)
in_dist_ds = in_dist_ds.remove_columns([x for x in in_dist_ds.column_names if x not in ['input_ids', "attention_mask"]])
in_dist_ds = in_dist_ds.select(range(NUM_SAMPLES_TO_TEST))
in_dist_test_dataloader = DataLoader(
        in_dist_ds, shuffle=False, collate_fn=default_data_collator, batch_size=BATCH_SIZE, pin_memory=True
)

#out-distribution test - toxic data (not used in training our detoxified model)
test_dataset_name = "OxAISH-AL-LLM/wiki_toxic"
out_dist_ds = load_dataset(test_dataset_name, split="test")
out_dist_ds = out_dist_ds.filter(lambda x: x["label"] == 1)
out_dist_ds = out_dist_ds.rename_columns({"comment_text":"text"})
def get_text2(x):
    x['text'] = x["text"][:context_length]
    return x
out_dist_ds = out_dist_ds.map(get_text2, batched=False)
out_dist_ds = tokenize_on_dataset(tokenizer, out_dist_ds)
out_dist_ds = out_dist_ds.remove_columns([x for x in out_dist_ds.column_names if x not in ['input_ids', "attention_mask"]])
out_dist_ds = out_dist_ds.select(range(NUM_SAMPLES_TO_TEST))
out_dist_test_dataloader = DataLoader(
        out_dist_ds, shuffle=False, collate_fn=default_data_collator, batch_size=BATCH_SIZE, pin_memory=True
)

#out-distribution test - benign data
benign_dataset_name = "wikitext"
benign_ds = load_dataset(benign_dataset_name, "wikitext-2-v1", split="test")
def get_text2(x):
    x['text'] = x["text"][:context_length]
    return x
benign_ds = benign_ds.map(get_text2, batched=False)
benign_ds = tokenize_on_dataset(tokenizer, benign_ds)
benign_ds = benign_ds.remove_columns([x for x in benign_ds.column_names if x not in ['input_ids', "attention_mask"]])
benign_ds = benign_ds.select(range(NUM_SAMPLES_TO_TEST))
benign_test_dataloader = DataLoader(
        benign_ds, shuffle=False, collate_fn=default_data_collator, batch_size=BATCH_SIZE, pin_memory=True
)

# evaluation setup
models_to_be_tested = [
    "./saved_model/EleutherAI/gpt-neo-125m_detoxified_jigsaw_unintended_bias",
    # "./saved_model/EleutherAI/gpt-neo-125m_detoxified",
    "EleutherAI/gpt-neo-125m",
    # "./saved_model/EleutherAI/gpt-neo-1.3b_detoxified",
    # "EleutherAI/gpt-neo-1.3b",
]
datasets_to_be_tested = {
    dataset_name: in_dist_test_dataloader,
    test_dataset_name: out_dist_test_dataloader,
    benign_dataset_name: benign_test_dataloader
}
output_file = f"./results/toxicity_{dataset_name.split('/')[-1]}.csv"
max_new_tokens = 50

# open a csv file for writing results
file = open(f"{output_file}", "a", newline="")
writer = csv.writer(file)
# add first rows
writer.writerow(["model_id", "dataset_id", "mean_toxicity", "std_toxicity"])

for model_id in tqdm(models_to_be_tested):
    model = None
    torch.cuda.empty_cache()
    toxicities = {}

    if "saved_model" in model_id: # detoxified model
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     device_map={"": device},
                                                    )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else: # base model (before detoxifying)
        model, tokenizer = load_pretrained_model_tokenizer(model_id, device=device, cache_dir=cache_dir)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for dataset_test in datasets_to_be_tested:
        ds_data_loader = datasets_to_be_tested[dataset_test]
        for inputs in ds_data_loader:
            inputs['input_ids'] = torch.tensor(inputs['input_ids']).to(device)
            inputs['attention_mask'] = torch.tensor(inputs['attention_mask']).to(device)
            seq_length = inputs['input_ids'].size(1)
            outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)
            generated_texts = tokenizer.batch_decode(outputs[:, seq_length:], skip_special_tokens=True)
            toxicity_score = toxicity.compute(predictions=generated_texts)

            if dataset_test not in toxicities:
                toxicities[dataset_test] = []
            toxicities[dataset_test].extend(toxicity_score["toxicity"])

        # compute mean & std using np
        mean = np.mean(toxicities[dataset_test])
        std = np.std(toxicities[dataset_test])

        # save to file
        writer.writerow([model_id, dataset_test, mean, std])

        # print
        print(f"Model: {model_id} - Dataset: {dataset_test} - Mean: {mean} - Std: {std}")

# close file
file.close()