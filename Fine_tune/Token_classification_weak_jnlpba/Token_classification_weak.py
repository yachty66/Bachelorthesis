from transformers import DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from datasets import DatasetDict, Dataset
import numpy as np
import matplotlib.pyplot as plt
import wandb
import random
import pandas as pd
import transformers

wandb.init(
    project="Bachelor_Thesis",
    entity="maxhager28",
    name="Token_classification_jnlpb_test_eval_test",
)
random.seed(42)

task = "ner"  # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "bert-base-cased"
batch_size = 8

jnlpba = load_dataset("jnlpba", split=["train[:100]", "validation[:100]"])
jnlpba = DatasetDict({"train": jnlpba[0], "validation": jnlpba[1]})

class JnlpbDataset:
    def __init__(self, dataset, portion, type_path):
        self.dataset = dataset[type_path]
        self.portion = portion
        self.remove()
        self.merge()
        self.apply()
        
    def remove_rows(self, row):
        ner_tags = row["ner_tags"]
        for i in range(len(ner_tags) - 1):
            if ner_tags[i] != 0 and ner_tags[i] == ner_tags[i + 1]:
                return False
        return True
                
    def remove(self):
        df = pd.DataFrame(self.dataset)
        df = df[df.apply(self.remove_rows, axis=1)]
        self.dataset = Dataset.from_pandas(df)

    def map_tags(self, row):
        mapping = {
            0: "O",
            1: "B-DNA",
            2: "I-DNA",
            3: "B-RNA",
            4: "I-RNA",
            5: "B-cell_line",
            6: "I-cell_line",
            7: "B-cell_type",
            8: "I-cell_type",
            9: "B-protein",
            10: "I-protein",
        }
        row["ner_tags"] = [[mapping[tag] for tag in row["ner_tags"]]][0]
        return row

    def merge_tags(self, tags, tokens):
        merged_tags = []
        merged_tokens = []
        i = 0
        while i < len(tags):
            if tags[i].startswith("B-"):
                merged_tag = tags[i][2:]
                merged_token = tokens[i]
                i += 1
                while i < len(tags) and tags[i].startswith("I-"):
                    merged_tag += " " + tags[i][2:]
                    merged_token += " " + tokens[i]
                    i += 1
                merged_tags.append(merged_tag)
                merged_tokens.append(merged_token)
            else:
                merged_tags.append(tags[i])
                merged_tokens.append(tokens[i])
                i += 1
        for i in range(len(merged_tags)):
            s = merged_tags[i].split()[0]
            s = s[0].upper() + s[1:]
            merged_tags[i] = s
        return merged_tags, merged_tokens

    def merge(self):
        df_train = pd.DataFrame(self.dataset)
        df_train = df_train.apply(self.map_tags, axis=1)
        df_train[["ner_tags", "tokens"]] = df_train.apply(
            lambda x: self.merge_tags(x["ner_tags"], x["tokens"]),
            axis=1,
            result_type="expand",
        )
        self.dataset = Dataset.from_pandas(df_train)

    def missing(self, row):
        lst = row["ner_tags"]
        if any(x != 0 for x in lst):
            index = random.choice([i for i, x in enumerate(lst) if x != 0])
            lst[index] = 0
            row["ner_tags"] = lst
            return row
        else:
            return row

    def wrong(self, row, num_tags):
        lst = row["ner_tags"]
        tags = []
        for i in range(1, num_tags):
            tags.append(i)
        if any(x != 0 for x in lst):
            indices = [i for i, x in enumerate(lst) if x != 0]
            random_index = random.choice(indices)
            current_value = lst[random_index]
            random_number = random.choice(
                [x for x in [1, 2, 3, 4, 5] if x != current_value]
            )
            lst[random_index] = random_number
            row["ner_tags"] = lst
            return row
        else:
            return row

    def apply(self):
        num_portion = int(len(self.dataset) * self.portion / 100)
        df = self.dataset.to_pandas()
        tags = [tag for row in df["ner_tags"] for tag in row]
        unique_tags = set(tags)
        mapping = {
            "O": 0,
            "RNA": 1,
            "DNA": 2,
            "Cell_line": 3,
            "Cell_type": 4,
            "Protein": 5,
        }
        df["ner_tags"] = [[mapping[tag] for tag in tags] for tags in df["ner_tags"]]
        for i in range(num_portion):
            random_number = random.randint(1, 2)
            if random_number == 1:
                new_row = self.missing(df.iloc[i])
                df.iloc[i] = new_row
            elif random_number == 2:
                num_tags = len(unique_tags)
                new_row = self.wrong(df.iloc[i], num_tags)
                df.iloc[i] = new_row
        self.dataset = Dataset.from_pandas(df)

    def get_dataset(self):
        return self.dataset


input_dataset_train = JnlpbDataset(dataset=jnlpba, portion=50, type_path="train")
dataset_train = input_dataset_train.get_dataset()
input_dataset_validation = JnlpbDataset(
    dataset=jnlpba, portion=0, type_path="validation"
)
dataset_validation = input_dataset_validation.get_dataset()
datasets = DatasetDict({"train": dataset_train, "validation": dataset_validation})
df = dataset_train.to_pandas()
label_list = list(set([tag for row in df["ner_tags"] for tag in row]))
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

label_all_tokens = True


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    #print(tokenizer.decode(tokenized_inputs["input_ids"][0]))
    #print labels 
    #print(examples["ner_tags"][0])
    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    tokenized_inputs["ner_tags"] = examples["ner_tags"]
    print(tokenized_inputs["labels"][0])
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list)
)

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-{task}-jnlpba-weak-labelled",
    evaluation_strategy="steps",
    eval_steps=1,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    # report_to="wandb",
    # push_to_hub=True,
    # todo hide token
    # push_to_hub_token="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"
)

data_collator = DataCollatorForTokenClassification(tokenizer)


def custom_compute_metrics(p, eval_dataset):
    predictions, labels = p
    original_predictions = []
    for i, label in enumerate(labels):
        corresponding_text = eval_dataset[i]["tokens"]
        original_predictions.append(corresponding_text)
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    '''print("true")
    print(true_labels)
    print("------")
    print("pred")
    print(true_predictions)
    print("------")'''
    true_predictions = np.array(
        [item for sublist in true_predictions for item in sublist]
    )
    true_labels = np.array([item for sublist in true_labels for item in sublist])
    cm = confusion_matrix(true_labels, true_predictions)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    mapping = {"O": 0, "rna": 1, "dna": 2, "cell_line": 3, "cell_type": 4, "protein": 5}
    reverse_mapping = {v: k for k, v in mapping.items()}
    ax = plt.gca()
    ax.set_xticks([i for i in range(len(mapping))])
    ax.set_yticks([i for i in range(len(mapping))])
    ax.set_xticklabels([reverse_mapping[i] for i in range(len(mapping))])
    ax.set_yticklabels([reverse_mapping[i] for i in range(len(mapping))])

    # wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.clf()
    accuracy = accuracy_score(true_labels, true_predictions)
    precision, recall, fscore, support = precision_recall_fscore_support(
        true_labels, true_predictions, zero_division=1, average="weighted"
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": fscore,
        "accuracy": accuracy,
        "support": support,
    }

trainer = Trainer(
    model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=lambda p: custom_compute_metrics(
        p, tokenized_datasets["validation"]
    ),
)

trainer.train()

wandb.finish()
