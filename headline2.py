import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
import re
from transformers import AdamWeightDecay
from datasets import Dataset
import math
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import TFAutoModelForSeq2SeqLM
from transformers.keras_callbacks import KerasMetricCallback



f = open('DryRun_Headline_Generation.json')
df = pd.read_json(f)
print(df.info())
df['news'] = df['news'].apply(lambda x: re.sub(r'\([^)]*\)', '', x))
#df['text'] = df[['news', 'headline']].apply(" ".join, axis=1)
#print(df['text'].head())
f.close()


dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)
for i in dataset['test']['headline']:
    print(len(i.split( )))
tokenizer = AutoTokenizer.from_pretrained("t5-small")

prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["news"]]
    model_inputs = tokenizer(inputs, max_length=2024, truncation=True)

    labels = tokenizer(text_target=examples["headline"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess_function, batched=True)
print(tokenized["train"][1])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small", return_tensors="tf")

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-small")

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

model.compile(optimizer=optimizer)

eval_loss = model.evaluate(tf_test_set)
print(f"Pretrained LM Perplexity: {math.exp(eval_loss):.2f}")

#metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_test_set)

model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=5)

eval_loss = model.evaluate(tf_test_set)
print(f"Finetuned Perplexity: {math.exp(eval_loss):.2f}")
'''
for i in range(len(dataset['test'])):
    text = prefix + dataset['test']['news'][i]
    inputs = tokenizer(text, return_tensors="tf").input_ids
    outputs = model.generate(inputs, max_new_tokens=len(dataset['test']['headline'][i]), do_sample=False)
    predictions = " ".join(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
    print()
    print(len(dataset['test']['headline'][i]))
    print(predictions)
    #print("News: ", dataset['test']['news'][0])
    answer = dataset['test']['headline'][i]
    print("Answer: ", answer)
    print(rouge.compute(predictions=[predictions], references=[answer]))
'''
for i in dataset['test']:
    text = prefix + i['news']
    inputs = tokenizer(text, return_tensors="tf").input_ids
    outputs = model.generate(inputs, max_new_tokens=len(i['headline'].split( )), do_sample=False)
    predictions = " ".join(tokenizer.batch_decode(outputs[0], skip_special_tokens=True)).strip()
    print()
    print(len(i['headline'].split( )))
    print(predictions)
    #print("News: ", dataset['test']['news'][0])
    answer = i['headline']
    print("Answer: ", answer)
    print(rouge.compute(predictions=[predictions], references=[answer]))