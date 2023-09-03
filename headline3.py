import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
import re
from transformers import AdamWeightDecay
from datasets import Dataset
import math
from transformers import T5ForConditionalGeneration
import evaluate
import numpy as np
from transformers import TFAutoModelForSeq2SeqLM
from transformers.keras_callbacks import KerasMetricCallback

# https://huggingface.co/JulesBelveze/t5-small-headline-generator

f = open('DryRun_Headline_Generation.json')
df = pd.read_json(f)
print(df.info())
df['news'] = df['news'].apply(lambda x: re.sub(r'\([^)]*\)', '', x)) #Remove Time stamps
df['headline'] = df['headline'].apply(lambda x: re.sub(r'[^\w\s]', '', x)) #Remove punctuation
#df['text'] = df[['news', 'headline']].apply(" ".join, axis=1)
#print(df['text'].head())
f.close()

test_str = "I ate 8 apple's puncak!"
res = re.sub(r'[^\w\s]', '', test_str)


dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)


#WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

rouge = evaluate.load("rouge")
model_name = "JulesBelveze/t5-small-headline-generator"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

for i in dataset['test']:
    input_ids = tokenizer(i['news'],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=3084)["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=128,
        no_repeat_ngram_size=2,
        num_beams=4)[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True)

    print(summary)
    print(i['headline'])
    print(rouge.compute(predictions=[summary], references=[i['headline']], use_stemmer=True))
    print("")

'''
# https://huggingface.co/learn/nlp-course/chapter7/5?fw=pt
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

def preprocess_function(examples):
    model_inputs = tokenizer(examples["news"], max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["headline"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess_function, batched=True)
print(tokenized["train"][1])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="google/mt5-small", return_tensors="tf")

rouge = evaluate.load("rouge")

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

model = TFAutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

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

for i in dataset['test']:
    inputs = tokenizer(i['news'], return_tensors="tf").input_ids
    outputs = model.generate(inputs, max_new_tokens=len(i['headline'].split( ))+3, do_sample=False)
    predictions = " ".join(tokenizer.batch_decode(outputs[0], skip_special_tokens=True)).strip()
    print()
    print(len(i['headline'].split( )))
    print(predictions)
    #print("News: ", dataset['test']['news'][0])
    answer = i['headline']
    print("Answer: ", answer)
    print(rouge.compute(predictions=[predictions], references=[answer]))

'''


