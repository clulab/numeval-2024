import json

f = open('DryRun_Numerical_Reasoning.json')

data = json.load(f)

print(data[0])
print("")
print(data[1])
print("")
print(data[2])

#for i in data:
#    print(i)

f.close()