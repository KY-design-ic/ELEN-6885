total_params = sum(p.numel() for p in model.parameters())
total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"#trainable parameters: {total_params_trainable}")
print(f"ratio of trainable parameters: {total_params_trainable/total_params}")

learning_rate = (1.47e-5) * 2
mini_batch_size = 64
ppo_epochs = 20
batch_size = 64

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

ppo_config = PPOConfig(
    #output_dir="./ppo_results",
    model_name=model_name_or_path,
    learning_rate=learning_rate,
    ppo_epochs=ppo_epochs,
    mini_batch_size=batch_size,
    batch_size=batch_size,
    gradient_accumulation_steps=1,
)
###model_name='EleutherAI/gpt-neo-125m'
#ppo_epochs=100


ppo_trainer = PPOTrainer(
    ppo_config,
    model,
    ref_model=reference_model,
    tokenizer=tokenizer,
    dataset=train_dataset,
    data_collator=lambda data: dict((key, [d[key] for d in data]) for key in data[0]),
    optimizer=optimizer,
)
#ref_model=reference_model
#dataset=train_dataset
#optimizer=optimizer

toxicity_model_name_or_path = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_name_or_path, cache_dir=cache_dir)
toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_name_or_path,
                                                                  cache_dir=cache_dir).to(ppo_trainer.accelerator.device)

max_new_tokens = 32
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": max_new_tokens,
}

model_save_path = f"./saved_model/{model_name_or_path}_detoxified_{dataset_name.split('/')[-1]}"

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_inputs = batch["input_ids"]

    # Get response from the policy model
    outputs_tensors = []
    for query in query_inputs:
        response = ppo_trainer.generate(query, **generation_kwargs)
        outputs_tensors.append(response.squeeze()[-max_new_tokens:])
    batch["output"] = [tokenizer.decode(r.squeeze()) for r in outputs_tensors]

    # Compute toxicity score for each output
    toxicity_inputs = toxicity_tokenizer(batch["output"], padding=True, truncation=True, return_tensors="pt")
    toxicity_inputs = toxicity_inputs.to(ppo_trainer.accelerator.device)
    logits = toxicity_model(**toxicity_inputs).logits.float()
    toxicity_labels = (logits[:, 0]).tolist()

    rewards = [torch.tensor(output) for output in toxicity_labels]

    # Run PPO optimization step
    stats = ppo_trainer.step(query_inputs, outputs_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 20 epochs
    if epoch and epoch % 20 == 0:
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(model_save_path)

import evaluate
import csv
import numpy as np
from torch.utils.data import DataLoader
from transformers import default_data_collator

## load toxicity from evaluation to score each model output
toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")

NUM_SAMPLES_TO_TEST = 500
BATCH_SIZE = 64
context_length = 500