
from datasets import load_dataset, load_metric
jnlpba = load_dataset('jnlpba', split=['train[:1]', "validation[:1]"])

print(jnlpba)