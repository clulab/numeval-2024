import json
from transformers import pipeline

f = open('DryRun_Numerical_Reasoning.json')

data = json.load(f)

print(data[0]['masked headline'].replace('____', '<mask> '))
print("")
print(data[1])
print("")
print(data[2])

#for i in data:
#    print(i)
f.close()

text = data[0]['masked headline'].replace('____', '<mask> ')

mask_filler = pipeline("fill-mask", "stevhliu/my_awesome_eli5_mlm_model")

print("")
print(mask_filler(text, top_k=3))