# Repository zur Bachelorarbeit "Named-Entity-Recognition mit schwach annotierten Daten". 

## Folgende Experimente aus der Arbeit können reproduziert werden:

- **Stark_BERT**

- **Schwach_50_BERT**

- **Schwach_100_BERT**

- **Stark_T5**

- **Schwach_50_T5**

- **Schwach_100_T5**

### Anleitung um BERT Experimente zu reproduzieren:

`BERT.py` in `Reproduktion/BERT/BERT.py` rennen lassen. Für das schwächen des Datensatzes kann in der Zeile

```python
input_dataset_train = JnlpbDataset(dataset=jnlpba, portion=0, type_path="train")
```

die `portion` gesetzt werden. `portion=0` bedeutet, dass der Datensatz nicht geschwächt wird. Bei `portion=50` werden 50% des Datensatzes geschwächt und bei `portion=100` werden 100% des Datensatzes geschwächt.

### Anforderungen BERT

`Reproduktion/BERT/requirements.txt`


### Anleitung um T5 Experimente zu reproduzieren:

`T5.py` in `Reproduktion/T5/T5.py` rennen lassen. Für das schwächen des Datensatzes kann in 

```python
def get_dataset(tokenizer, type_path, args):
        tokenizer.max_length = args.max_seq_length
        tokenizer.model_max_length = args.max_seq_length
        jnlpba = load_dataset("jnlpba", split=["train[:18500]", "validation[:3500]"])
        jnlpba = DatasetDict({"train": jnlpba[0], "validation": jnlpba[1]})
        dataset = jnlpba
        return JnlpbDataset(
            tokenizer=tokenizer, dataset=dataset, type_path=type_path, portion=0
        )
```

Die Zeile an der Stelle `portion` eingestellt werden.

```python
return JnlpbDataset(
            tokenizer=tokenizer, dataset=dataset, type_path=type_path, portion=0
        )
```

`portion=0` bedeutet, dass der Datensatz nicht geschwächt wird. Bei `portion=50` werden 50% des Datensatzes geschwächt und bei `portion=100` werden 100% des Datensatzes geschwächt.

### Anforderungen T5

`Reproduktion/T5/requirements.txt`