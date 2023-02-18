from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset, load_metric
from datasets import DatasetDict
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch
import torchmetrics
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
import wandb
from wandb import AlertLevel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

os.environ["WANDB_API_KEY"] = "8ba76c3e9dc8954082ddbb65ca4b5c94ea6ac81a"

# --------- Train args
args_dict = dict(
        data_dir="jnlpba",
        output_dir="checkpoints",
        model_name_or_path="t5-small",
        tokenizer_name_or_path="t5-small",
        max_seq_length=256,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=8,
        eval_batch_size=8,
        num_train_epochs=1,
        gradient_accumulation_steps=16,
        n_gpu=1,
        early_stop_callback=False,
        fp_16=True,
        opt_level="O1",
        max_grad_norm=1,
        seed=42,
        val_check_interval=0.33,
    )
# ---------------------

import nltk
nltk.download('punkt')

# TODO
wandb.login()
wandb.init(project="Bachelor_Thesis", entity="maxhager28",
           name="Seq2seq_jnlpba_weak_50_18500")

#wandb.init(mode="disabled")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# --------- MODEL -----------
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparam):
        super(T5FineTuner, self).__init__()
        self.hparam = hparam
        self.model = T5ForConditionalGeneration.from_pretrained(
            hparam.model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(hparam.model_name_or_path)
        self.save_hyperparameters()
        self.true = []
        self.pred = []
        self.batch_counter = 0
        # self.counter = 0

    def is_logger(self):
        return True

    def label_true(self, incoming, actual):
        l_targets = [
            [tuple_list[0] for tuple_list in sublist] for sublist in actual
        ]
        l_predictions = []
        for x in incoming:
            result = re.split(";(?![^\(]*\))", x)
            result = [x.strip() for x in result]
            l_predictions.append(
                [{e.split(":")[0].strip(): e.split(":")[1].strip()} for e in result if e])
        result = []
        for inner_list in l_targets:
            outcome_inner = []
            for word in inner_list:
                found = False
                for dict_list in l_predictions:
                    for dict_item in dict_list:
                        if word.lower() in dict_item.values():
                            outcome_inner.append(list(dict_item.keys())[0])
                            found = True
                            break
                    if found:
                        break
                if not found:
                    outcome_inner.append("O")
            result.append(outcome_inner)
        print("label true inside")
        print("incoming")
        print(incoming)
        print(30 * "-")
        print("l_targets")
        print(l_targets)
        print(30 * "-")
        print("l_predictions")
        print(l_predictions)
        print(30 * "-")
        print("result")
        print(result)
        print(30 * "-")
        return result

    def label_pred(self, incoming, actual):
        l_targets = [
            [tuple_list[0] for tuple_list in sublist] for sublist in actual
        ]
        l_predictions = []
        for string in incoming:
            matches = [
                match
                for match in re.findall(
                    r"(rna: (.+?))(;|$)|(dna: (.+?))(;|$)|(cell_line: (.+?))(;|$)|(protein: (.+?))(;|$)|(cell_type: (.+?))(;|$)",
                    string,
                )
                if match[1] or match[4] or match[7] or match[10] or match[13]
            ]
            inner_list = []
            for match in matches:
                if match[1]:
                    inner_list.append({"rna": match[1]})
                if match[4]:
                    inner_list.append({"dna": match[4]})
                if match[7]:
                    inner_list.append({"cell_line": match[7]})
                if match[10]:
                    inner_list.append({"protein": match[10]})
                if match[13]:
                    inner_list.append({"cell_type": match[13]})
            l_predictions.append(inner_list)

        result = []
        for inner_list in l_targets:
            outcome_inner = []
            for word in inner_list:
                found = False
                for dict_list in l_predictions:
                    for dict_item in dict_list:
                        if word.lower() in dict_item.values():
                            outcome_inner.append(list(dict_item.keys())[0])
                            found = True
                            break
                    if found:
                        break
                if not found:
                    outcome_inner.append("O")
            result.append(outcome_inner)
        return result

    def forward(
            self,
            input_ids,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            lm_labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss)
        wandb.log({"train_loss_step": loss})
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        wandb.log({"avg_train_loss": avg_train_loss})

    def map_tags(self, lst):
        mapping = {
            "O": 0,
            "rna": 1,
            "dna": 2,
            "cell_line": 3,
            "cell_type": 4,
            "protein": 5,
        }
        result = [[mapping[tag] for tag in tags] for tags in lst]
        return result

    def val_preprocessing(self, true, pred):
        new_true = []
        new_pred = []
        for i in range(len(true)):
            if true[i] == 0 and pred[i] == 0:
                continue
            else:
                new_true.append(true[i])
                new_pred.append(pred[i])
        return new_true, new_pred

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.batch_counter = 0
        print("batch ids")
        print(batch_idx)
        print(30 * "-")
        outputs = []
        targets = []
        all_text = []
        true_labels = []
        pred_labels = []
        predictions = []
        predictions_temp = []
        l_true_labels = []
        l_pred_labels = []
        input_ids = batch["source_ids"].to("cuda")
        attention_mask = batch["source_mask"].to("cuda")
        outs = model.model.generate(
            input_ids=input_ids, attention_mask=attention_mask
        )
        dec = [
            tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            for ids in outs
        ]
        print("dec")
        print(dec)
        print(30 * "-")
        target = [
            tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            for ids in batch["target_ids"]
        ]
        print("target")
        print(target)
        print(30 * "-")
        texts = [
            tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            for ids in batch["source_ids"]
        ]
        print("text not stripped")
        text = [
            tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            for ids in batch["source_ids"]
        ]
        print(text)
        print(30 * "-")
        print("text")
        print(texts)
        print(30 * "-")
        print("token length")
        print(len(batch["tokens"]))
        len_source_ids = len(batch["source_ids"])
        print("tokens length")
        print(len(batch["tokens"][self.batch_counter: self.batch_counter + len_source_ids]))
        print("source ids length")
        print(len(batch["source_ids"]))
        print("target_ids length")
        print(len(batch["target_ids"]))
        print("self batch counter")
        print(self.batch_counter)
        print("self batch counter + len source ids")
        print(self.batch_counter + len_source_ids)
        print(30 * "-")
        true_label = self.label_true(target, batch["tokens"][
                                             self.batch_counter: self.batch_counter + len_source_ids])
        predicted_label = self.label_pred(dec, batch["tokens"][
                                               self.batch_counter: self.batch_counter + len_source_ids])
        self.batch_counter += len_source_ids
        # self.counter += self.hparam.eval_batch_size
        pred_mapped = self.map_tags(predicted_label)
        true_mapped = self.map_tags(true_label)
        self.true.extend(np.array(true_mapped).flatten())
        self.pred.extend(np.array(pred_mapped).flatten())
        val_loss = self._step(batch)
        self.log("val_loss", val_loss)
        ##################################################################
        print("true_label")
        print(true_label)
        print("predicted_label")
        print(predicted_label)
        if true_label == [] and predicted_label == []:
            return
        true_label = np.concatenate(true_mapped)
        predicted_label = np.concatenate(pred_mapped)
        print("true_label epoch end")
        print(true_label)
        print(30 * "-")
        print("predicted_label epoch end")
        print(predicted_label)
        print(30 * "-")
        true_label, predicted_label = self.val_preprocessing(true_label, predicted_label)
        print("processed rue_label epoch end")
        print(true_label)
        print(30 * "-")
        print("processed predicted_label epoch end")
        print(predicted_label)
        print(30 * "-")
        cm = confusion_matrix(true_label, predicted_label)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        mapping = {
            "O": 0,
            "rna": 1,
            "dna": 2,
            "cell_line": 3,
            "cell_type": 4,
            "protein": 5,
        }
        reverse_mapping = {v: k for k, v in mapping.items()}
        ax = plt.gca()
        ax.set_xticks([i for i in range(len(mapping))])
        ax.set_yticks([i for i in range(len(mapping))])
        ax.set_xticklabels([reverse_mapping[i] for i in range(len(mapping))])
        ax.set_yticklabels([reverse_mapping[i] for i in range(len(mapping))])
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.clf()
        accuracy = accuracy_score(true_label, predicted_label)
        precision, recall, fscore, support = precision_recall_fscore_support(
            true_label, predicted_label, zero_division=1, average="weighted"
        )
        wandb.log(
            {
                "precision": precision,
                "recall": recall,
                "f1": fscore,
                "accuracy": accuracy,
            }
        )
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        pass
        '''self.counter = 0
        true_label = np.concatenate(self.true)
        predicted_label = np.concatenate(self.pred)
        print("true_label epoch end")
        print(true_label)
        print(30*"-")
        print("predicted_label epoch end")
        print(predicted_label)
        print(30*"-")
        true_label, predicted_label = self.val_preprocessing(true_label, predicted_label)
        print("processed rue_label epoch end")
        print(true_label)
        print(30*"-")
        print("processed predicted_label epoch end")
        print(predicted_label)
        print(30*"-")
        cm = confusion_matrix(true_label, predicted_label)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        mapping = {
            "O": 0,
            "rna": 1,
            "dna": 2,
            "cell_line": 3,
            "cell_type": 4,
            "protein": 5,
        }
        reverse_mapping = {v: k for k, v in mapping.items()}
        ax = plt.gca()
        ax.set_xticks([i for i in range(len(mapping))])
        ax.set_yticks([i for i in range(len(mapping))])
        ax.set_xticklabels([reverse_mapping[i] for i in range(len(mapping))])
        ax.set_yticklabels([reverse_mapping[i] for i in range(len(mapping))])
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.clf()
        accuracy = accuracy_score(true_label, predicted_label)
        precision, recall, fscore, support = precision_recall_fscore_support(
            true_label, predicted_label, zero_division=1, average="weighted"
        )
        wandb.log({'precision': precision, 'recall': recall, 'f1': fscore, 'accuracy': accuracy})'''

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparam.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparam.learning_rate,
            eps=self.hparam.adam_epsilon,
        )
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(
            self,
            epoch=None,
            batch_idx=None,
            optimizer=None,
            optimizer_idx=None,
            optimizer_closure=None,
            on_tpu=None,
            using_native_amp=None,
            using_lbfgs=None,
    ):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {
            "loss": "{:.3f}".format(self.trainer.avg_loss),
            "lr": self.lr_scheduler.get_last_lr()[-1],
        }
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="train", args=self.hparam
        )
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.hparam.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=2,
        )
        t_total = (
                (
                        len(dataloader.dataset)
                        // (
                                self.hparam.train_batch_size
                                * max(1, self.hparam.n_gpu if torch.cuda.is_available() else 1)
                        )
                )
                // self.hparam.gradient_accumulation_steps
                * float(self.hparam.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparam.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="validation",
                                  args=self.hparam)
        dataloader = DataLoader(val_dataset, batch_size=self.hparam.eval_batch_size, num_workers=1)
        return dataloader

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(
                pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(
                            key, str(metrics[key])))


from datasets import load_dataset, load_metric
from datasets import DatasetDict, Dataset
import random
import pandas as pd
import re

random.seed(42)


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
        # self.tokens = []
        self.remove()
        self.merge()
        self.convert()
        self.apply()
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        # tokens = self.tokens[index]["tokens"].squeeze()

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

    def remove(self):
        df = pd.DataFrame(self.dataset)
        df = df[df["tokens"].apply(lambda x: ";" not in x)]
        self.dataset = df

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
            # print(self.dataset)
            # print(30*".")
            # print(self.dataset[idx]["tokens"])
            input_, target = " ".join(self.dataset[idx]["tokens"]), "; ".join(
                self.dataset[idx]["spans"]
            )
            # tokens = self.dataset[idx]["tokens"] + ["</s>"]
            input_ = input_.lower() + " </s>"
            target = target.lower() + " </s>"
            # print(input_)
            # print(30*"-")
            # print(target)
            # print(30*"-")
            # print(tokens)
            # what are inputs and targets?
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
            """tokenized_tokens = self.tokenizer.batch_encode_plus(
                tokens,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )  """
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
            # self.tokens.append(tokenized_tokens)

    def missing(self, row):
        lst = row["ner_tags"]
        lst_spans = row["spans"]
        if any(x != 0 for x in lst):
            index = random.choice([i for i, x in enumerate(lst) if x != 0])
            index_to_remove = len([x for x in lst[:index] if x != 0])
            lst_spans = np.delete(lst_spans, index_to_remove)
            lst[index] = 0
            row["ner_tags"] = lst
            row["spans"] = lst_spans
            return row
        else:
            return row

    def wrong(self, row, num_tags):
        mapping = {
            0:"O",
            1:"RNA",
            2:"DNA",
            3:"Cell_line",
            4:"Cell_type",
            5:"Protein"
        }
        lst = row["ner_tags"]
        lst_spans = row["spans"]
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
            index_to_wrong = len([x for x in lst[:random_index] if x != 0])
            splitted = lst_spans[index_to_wrong].split(":")
            new_elem = mapping[random_number] + ":" + splitted[1]
            lst_spans[index_to_wrong] = new_elem
            row["spans"] = lst_spans
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

tokenizer = AutoTokenizer.from_pretrained("t5-small")
args = argparse.Namespace(**args_dict)
model = T5FineTuner(args)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="max",
        save_on_train_epoch_end=True
    )

train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        accelerator='gpu',
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        # early_stop_callback=False,
        precision=32,
        # amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        # checkpoint_callback=checkpoint_callback,
        # logger=wandb_logger,
        callbacks=[checkpoint_callback, LoggingCallback()],
    )

def get_dataset(tokenizer, type_path, args):
    tokenizer.max_length = args.max_seq_length
    tokenizer.model_max_length = args.max_seq_length
    jnlpba = load_dataset("jnlpba", split=["train[:18500]", "validation[:3500]"])
    jnlpba = DatasetDict({"train": jnlpba[0], "validation": jnlpba[1]})
    dataset = jnlpba
    return JnlpbDataset(
        tokenizer=tokenizer, dataset=dataset, type_path=type_path, portion=50
    )

trainer = pl.Trainer(**train_params)
trainer.fit(model)
wandb.save('lightning_logs/version_0/checkpoints/*ckpt*')
wandb.alert(
    title="End of training.",
    text="Training finished successfully.",
)
wandb.finish()
