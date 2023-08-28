import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
import re
from transformers import DataCollatorForLanguageModeling
from transformers import AdamWeightDecay
from transformers import TFAutoModelForCausalLM
from datasets import Dataset
import math


f = open('DryRun_Headline_Generation.json')
df = pd.read_json(f)
print(df.info())
df['news'] = df['news'].apply(lambda x: re.sub(r'\([^)]*\)', '', x))
df['text'] = df[['news', 'headline']].apply(" ".join, axis=1)
print(df['text'].head())
f.close()

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples['text']])

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenized = dataset.map(preprocess_function, batched=True, num_proc=4,
                        remove_columns=dataset["train"].column_names)
print(tokenized["train"][1])
block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized.map(group_texts,  batched=True, batch_size=1000, num_proc=4)

print('check', tokenizer.decode(lm_dataset["train"][1]["input_ids"]))
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

model = TFAutoModelForCausalLM.from_pretrained("distilgpt2")

tf_train_set = model.prepare_tf_dataset(
    lm_dataset["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    lm_dataset["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

model.compile(optimizer=optimizer, jit_compile=True)

eval_loss = model.evaluate(tf_test_set)
print(f"Pretrained LM Perplexity: {math.exp(eval_loss):.2f}")

model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=10)

eval_loss = model.evaluate(tf_test_set)
print(f"Finetuned Perplexity: {math.exp(eval_loss):.2f}")

prompt = dataset['test']['news'][0]
inputs = tokenizer(prompt, return_tensors="tf").input_ids
outputs = model.generate(input_ids=inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

print("News: ", dataset['test']['news'][0])
print("Answer: ", dataset['test']['headline'][0])
