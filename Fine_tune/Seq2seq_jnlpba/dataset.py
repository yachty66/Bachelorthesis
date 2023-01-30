

from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, load_metric
from datasets import DatasetDict
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
import random
from itertools import chain
from string import punctuation

# import wandb
# from wandb import AlertLevel
from pytorch_lightning import Trainer

# from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset, load_metric
from datasets import DatasetDict, Dataset
import random
import pandas as pd
import nltk

class JnlpbDataset(Dataset):
    def __init__(self, tokenizer, dataset, type_path, portion, max_len=512):
        self.dataset = dataset[type_path]
        self.portion = portion
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.tokenizer.max_length = max_len
        self.tokenizer.model_max_length = max_len
        self.inputs = []
        self.targets = []
        #self.tokens = []
        self.merge()
        self.convert()
        self.apply()
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        #tokens = self.tokens[index]["tokens"].squeeze()
        
        src_mask = self.inputs[index][
            "attention_mask"
        ].squeeze()  # might need to squeeze
        target_mask = self.targets[index][
            "attention_mask"
        ].squeeze()  # might need to squeeze
        tokens = self.dataset["tokens"]
        
        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "tokens": tokens,
        }

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

    def convert(self):
        df_train = pd.DataFrame(self.dataset)
        l = []
        l_temp = []
        for i in range(len(df_train)):
            for j in range(len(df_train["ner_tags"][i])):
                if df_train["ner_tags"][i][j] != "O":
                    l_temp.append(
                        df_train["ner_tags"][i][j] + ": " + df_train["tokens"][i][j]
                    )
            l.append(l_temp)
            l_temp = []
        d = {"spans": l}
        df_train = df_train.assign(spans=l)
        train = Dataset.from_pandas(df_train)
        self.dataset = train
        return train

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

    def _build(self):
        for idx in range(len(self.dataset)):
            #print(self.dataset)
            #print(30*".")
            #print(self.dataset[idx]["tokens"])
            input_, target = " ".join(self.dataset[idx]["tokens"]), "; ".join(
                self.dataset[idx]["spans"]
            )
            #tokens = self.dataset[idx]["tokens"] + ["</s>"]
            input_ = input_.lower() + " </s>"
            target = target.lower() + " </s>"
            #print(input_)
            #print(30*"-")
            #print(target)
            #print(30*"-")
            #print(tokens)
            #what are inputs and targets? 
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            '''tokenized_tokens = self.tokenizer.batch_encode_plus(
                tokens,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )  '''     
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
            #self.tokens.append(tokenized_tokens)

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

    def uncomplete(self):
        pass

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
            """else:
                    self.uncomplete()"""
        self.dataset = Dataset.from_pandas(df)

    def get_dataset(self):
        return self.dataset
