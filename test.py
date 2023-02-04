from pympler import asizeof
from datasets import load_dataset, load_metric
from datasets import DatasetDict

jnlpba = load_dataset("jnlpba", split=["train", "validation"])
jnlpba = DatasetDict({"train": jnlpba[0], "validation": jnlpba[1]})



counter = 0
#if a element in "tokens" list contains ";" print line
for i in range(0, len(jnlpba["train"])):
    for j in range(0, len(jnlpba["train"][i]["tokens"])):
        if ";" in jnlpba["train"][i]["tokens"][j]:
            #print(jnlpba["validation"][i])
            counter = counter + 1
            break
        
print(counter)