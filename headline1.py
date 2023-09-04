import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
import re
from transformers import AdamWeightDecay
from datasets import Dataset
import math
from transformers import DataCollatorForSeq2Seq
import evaluate
from transformers import PegasusForConditionalGeneration
from transformers import TFAutoModelForSeq2SeqLM
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

# https://huggingface.co/docs/transformers/tasks/summarization
# https://huggingface.co/learn/nlp-course/chapter7/5?fw=pt

f = open('DryRun_Headline_Generation.json')
df = pd.read_json(f)
print(df.info())
df['news'] = df['news'].apply(lambda x: re.sub(r'\([^)]*\)', '', x)) #Remove Time stamps
#df['headline'] = df['headline'].apply(lambda x: re.sub(r'[^\w\s]', '', x)) #Remove punctuation
f.close()

model_names = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)
#for i in dataset['test']['news']:
#    print(i)

rouge = evaluate.load("rouge")

prefix = "summarize: "
prefix_michau = "headline: "
max_len = 3024
max_l = 1028

def t5_model(examples, model_name, learning_rate, epochs=5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["news"]]
        model_inputs = tokenizer(inputs, max_length=max_len, truncation=True)
        labels = tokenizer(text_target=examples["headline"], max_length=max_l, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           model=model_name,
                                           return_tensors="tf")

    # try learning rate of 1e-4 also
    optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=0.01)

    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

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

    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=epochs)

    eval_loss = model.evaluate(tf_test_set)
    print(f"Finetuned Perplexity: {math.exp(eval_loss):.2f}")
    return tokenizer, model

def t5_predictions(data, tokenizer, model):
    text = prefix + data['news']
    inputs = tokenizer(text, return_tensors="tf").input_ids
    outputs = model.generate(inputs, max_new_tokens=len(i['headline'].split( ))+10, do_sample=False)
    predictions = " ".join(tokenizer.batch_decode(outputs[0], skip_special_tokens=True)).strip()
    return predictions


#t5_small_t, t5_small_m = t5_model(dataset, model_names[0], 1e-4)
t5_base_t, t5_base_m = t5_model(dataset, model_names[1], 1e-4)
#t5_small1_t, t5_small1_m = t5_model(dataset, model_names[0], 3e-4)
#t5_base1_t, t5_base1_m = t5_model(dataset, model_names[1], 3e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_michau = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
tokenizer_michau = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
model = model_michau.to(device)

model_jules = "JulesBelveze/t5-small-headline-generator"
tokenizer_jules = AutoTokenizer.from_pretrained(model_jules)
model_jules = T5ForConditionalGeneration.from_pretrained(model_jules)

tokenizer_pegasus = AutoTokenizer.from_pretrained("google/pegasus-xsum")
model_pegasus = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

for i in dataset['test']:
    '''
    text_t5 = prefix + i['news']
    inputs_t5 = tokenizer_t5_small(text_t5, return_tensors="tf").input_ids
    outputs_t5 = model_t5_small.generate(inputs_t5, max_new_tokens=len(i['headline'].split( ))+10, do_sample=False)
    predictions_t5 = " ".join(tokenizer_t5_small.batch_decode(outputs_t5[0], skip_special_tokens=True)).strip()
    '''
    #t5_small = t5_predictions(i, t5_small_t, t5_small_m)
    t5_base = t5_predictions(i, t5_base_t, t5_base_m)
    #t5_small1 = t5_predictions(i, t5_small1_t, t5_small1_m)
    #t5_base1 = t5_predictions(i, t5_base1_t, t5_base1_m)
    text_michau = prefix_michau + i['news']
    encoding_michau = tokenizer_michau.encode_plus(text_michau, return_tensors = "pt")
    input_ids_michau = encoding_michau["input_ids"].to(device)
    attention_masks_michau = encoding_michau["attention_mask"].to(device)
    outputs_michau = model_michau.generate(input_ids = input_ids_michau,
                                    attention_mask = attention_masks_michau,
    max_length = 64, num_beams = 3, early_stopping = True)
    result_michau = tokenizer_michau.decode(outputs_michau[0])
    result_michau = re.sub("\<.*?\>","", result_michau)

    input_ids_jules = tokenizer_jules(i['news'],
                              return_tensors="pt",
                              padding="max_length",
                              truncation=True,
                              max_length=3084)["input_ids"]

    output_ids_jules = model_jules.generate(input_ids=input_ids_jules,
            max_length=128, no_repeat_ngram_size=2, num_beams=4)[0]

    summary_jules = tokenizer_jules.decode(output_ids_jules,
            skip_special_tokens=True, clean_up_tokenization_spaces=True)

    inputs_pegasus = tokenizer_pegasus(i['news'], max_length=3024, return_tensors="pt")
    summary_ids_pegasus = model_pegasus.generate(inputs_pegasus["input_ids"])
    summary_pegasus = tokenizer_pegasus.batch_decode(summary_ids_pegasus, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    answer = i['headline']
    print("Answer: ", answer)
    #print("T5 small: ", t5_small)
    #print("T5_small: ", rouge.compute(predictions=[t5_small], references=[answer]))

    print("T5_base: ", t5_base)
    print("T5_base : ", rouge.compute(predictions=[t5_base], references=[answer]))
    '''
    print("T5 small 1: ", t5_small1)
    print("T5_small 1: ", rouge.compute(predictions=[t5_small1], references=[answer]))
    print("T5_base 1: ", t5_base1)
    print("T5_base 1: ", rouge.compute(predictions=[t5_base1], references=[answer]))
    '''
    print("Michau: ", result_michau)
    print("Michau: ", rouge.compute(predictions=[result_michau], references=[answer]))
    print("Jules: ", summary_jules)
    print("Jules: ", rouge.compute(predictions=[summary_jules], references=[i['headline']],
                        use_stemmer=True))
    print("Pegasus: ", summary_pegasus)
    print("Pegasus: ", rouge.compute(predictions=[summary_pegasus], references=[i['headline']], use_stemmer=True))
    print("")

