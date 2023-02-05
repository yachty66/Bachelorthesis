from pympler import asizeof
import pandas as pd
from datasets import load_dataset, load_metric
from datasets import DatasetDict

jnlpba = load_dataset("jnlpba", split=["train[:100]", "validation[:100]"])
jnlpba = DatasetDict({"train": jnlpba[0], "validation": jnlpba[1]})


#print first row of train
print(jnlpba["train"][0])

#df = pd.DataFrame(jnlpba["validation"])
#df = df[~df["tokens"].str.contains(";")]
#df = df[df['tokens'].apply(lambda x: ";" not in x)]
#df_filtered = df[df['tokens'].apply(lambda x: ";" in x)]


#get the first element of df_filtered
#print(df_filtered.iloc[0]["tokens"])

#print the element which contains ";" inside od df_filtered.iloc[0]["tokens"]
#print(df_filtered.iloc[0]["tokens"][df_filtered.iloc[0]["tokens"].index(";")])
