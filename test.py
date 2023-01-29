from datasets import load_dataset, DatasetDict

jnlpba = load_dataset("jnlpba", split=["train[:10]", "validation[:10]"])
jnlpba = DatasetDict({"train": jnlpba[0], "validation": jnlpba[1]})