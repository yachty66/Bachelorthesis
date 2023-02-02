from datasets import load_dataset, load_metric
from datasets import DatasetDict, Dataset
import pandas as pd
#load dataset  
#check for how many rows its true that two numbers are side by side and the same number

jnlpba = load_dataset("jnlpba", split=["train[:1000]", "validation[:1000]"])
jnlpba = DatasetDict({"train": jnlpba[0], "validation": jnlpba[1]})
counter = 0
 
 
df = pd.DataFrame(jnlpba["train"])
for i in range (len(df)):
    for j in range(len(df["ner_tags"][i]) - 1):
        if df["ner_tags"][i][j] != 0 and df["ner_tags"][i][j] == df["ner_tags"][i][j + 1]:
            counter += 1
            break 
            #print(jnlpba["train"]["ner_tags"][i])
            #print("True")
    
    



print(counter)