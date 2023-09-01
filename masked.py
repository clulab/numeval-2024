import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
import re
from transformers import DataCollatorForLanguageModeling
from transformers import AdamWeightDecay
from transformers import TFAutoModelForMaskedLM
from datasets import Dataset
import math

#https://huggingface.co/docs/transformers/tasks/masked_language_modeling#inference

f = open('DryRun_Numerical_Reasoning.json')
df = pd.read_json(f)
print(df.head())
df['news'] = df['news'].apply(lambda x: re.sub(r'\([^)]*\)', '', x))
df['masked headline'] = df['masked headline'].str.replace('____', '<mask> ')
df['text'] = df[['news', 'masked headline']].apply(" ".join, axis=1)
print(df['text'].head())
#df['ans']=df['ans'].str.replace(',','')
f.close()

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

def preprocess_function(examples):
    #return tokenizer([" ".join(x) for x in examples['text']]) # not join
    return tokenizer(examples['text'])

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", use_fast=False)
tokenized = dataset.map(preprocess_function, batched=True, num_proc=4,
                        remove_columns=dataset["train"].column_names)

block_size = 128

def group_texts(examples): #skipp
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

#lm_dataset = tokenized.map(group_texts,  batched=True, num_proc=4)

''' DO I NEED TO SPECIFY mlm_probability SINCE THE MASKED HEADLINE IS ALREAYD MASKED? '''
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, return_tensors="tf")

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

model = TFAutoModelForMaskedLM.from_pretrained("distilroberta-base")

'''
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
'''
tf_train_set = model.prepare_tf_dataset(
    tokenized["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    tokenized["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

model.compile(optimizer=optimizer)  # No loss argument!

eval_loss = model.evaluate(tf_test_set)
print(f"Pretrained LM Perplexity: {math.exp(eval_loss):.2f}")

'''
checkpoint_path = "training_1/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
'''

model.fit(x=tf_train_set, epochs=15)

eval_loss = model.evaluate(tf_test_set)
print(f"Finetuned Perplexity: {math.exp(eval_loss):.2f}")

'''
test_sentence = dataset['test']['text'][1]
inputs = tokenizer(test_sentence, return_tensors="tf")
mask_token_index = tf.where(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
logits = model(**inputs).logits
mask_token_logits = logits[0, mask_token_index, :]

top_3_tokens = tf.math.top_k(mask_token_logits, 3).indices.numpy()

for token in top_3_tokens:
    print(tokenizer.decode([token]))
    print(test_sentence.replace(tokenizer.mask_token, tokenizer.decode([token])))
print("Answer: ", dataset['test']['ans'][1])
print(dataset['test'][1])
'''
'''
mask_filler = pipeline(
    "fill-mask", model="training_1/cp.ckpt")

print(dataset['test']['masked headline'][0])
print(mask_filler("The most common household pets are <mask> and dogs.", top_k=1))

print(mask_filler(dataset['test']['masked headline'][0], top_k=3))
'''
for i in range(len(dataset['test'])):
    inputs = tokenizer(dataset['test']['text'][i], return_tensors="tf")
    mask_token_index = tf.where(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
    logits = model(**inputs).logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_token = tf.math.top_k(mask_token_logits, 1).indices.numpy()
    prediction = tokenizer.decode([top_token][0]).strip()
    print("Prediction: ", prediction)
    print("Answer: ", dataset['test']['ans'][i])
    print(prediction == dataset['test']['ans'][i])
