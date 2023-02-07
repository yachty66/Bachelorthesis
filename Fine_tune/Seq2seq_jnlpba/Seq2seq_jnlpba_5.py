if __name__ == "__main__":
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
    import torch
    import numpy as np
    import pandas as pd
    from nltk.tokenize import sent_tokenize
    import argparse
    import os
    import logging
    import random
    import re
    from itertools import chain
    from string import punctuation

    import wandb
    from wandb import AlertLevel
    from pytorch_lightning import Trainer

    # from pytorch_lightning.loggers import WandbLogger
    from datasets import load_dataset, load_metric
    from datasets import DatasetDict, Dataset
    import random
    import pandas as pd
    import nltk
    from dataset import JnlpbDataset

    # nltk.download("punkt")
    random.seed(42)

    wandb.init(
        project="Bachelor_Thesis",
        entity="maxhager28",
        name="Seq2seq_jnlpba_strong_test_test",
    )

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(42)

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
            self.counter = 0

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
                    [
                        {e.split(":")[0].strip(): e.split(":")[1].strip()}
                        for e in result
                        if e
                    ]
                )
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
            input_ids = batch["source_ids"].to("cpu")
            attention_mask = batch["source_mask"].to("cpu")
            outs = model.model.generate(
                input_ids=input_ids, attention_mask=attention_mask
            )
            dec = [
                tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ).strip()
                for ids in outs
            ]
            target = [
                tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ).strip()
                for ids in batch["target_ids"]
            ]
            texts = [
                tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ).strip()
                for ids in batch["source_ids"]
            ]
            true_label = self.label_true(
                target,
                batch["tokens"][
                    self.counter : (self.hparam.eval_batch_size + self.counter)
                ],
            )
            predicted_label = self.label_pred(
                dec,
                batch["tokens"][
                    self.counter : (self.hparam.eval_batch_size + self.counter)
                ],
            )
            self.counter += self.hparam.eval_batch_size
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
            true_label, predicted_label = self.val_preprocessing(
                true_label, predicted_label
            )
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
            
            
            
            
            
                       
            return true_mapped, pred_mapped

        def validation_epoch_end(self, outputs):
            true_label = np.concatenate(self.true)
            predicted_label = np.concatenate(self.pred)
            true_label, predicted_label = self.val_preprocessing(
                true_label, predicted_label
            )
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
            val_dataset = get_dataset(
                tokenizer=self.tokenizer, type_path="validation", args=self.hparam
            )
            dataloader = DataLoader(
                val_dataset, batch_size=self.hparam.eval_batch_size, num_workers=2
            )
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
                    pl_module.hparams.output_dir, "test_results.txt"
                )
                with open(output_test_results_file, "w") as writer:
                    for key in sorted(metrics):
                        if key not in ["log", "progress_bar"]:
                            logger.info("{} = {}\n".format(key, str(metrics[key])))
                            writer.write("{} = {}\n".format(key, str(metrics[key])))

    args_dict = dict(
        data_dir="jnlpba",  # path for data files
        output_dir="checkpoints",  # path to save the checkpoints
        model_name_or_path="t5-small",
        tokenizer_name_or_path="t5-small",
        max_seq_length=512,  # todo figure out
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=8,  # 4/2/1 if t5-small not working
        eval_batch_size=8,
        num_train_epochs=1,
        gradient_accumulation_steps=16,
        # n_gpu=1,
        early_stop_callback=False,
        fp_16=True,  # if you want to enable 16-bit training then install apex and set this to true
        opt_level="O1",  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
        val_check_interval=0.33,
    )

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    args = argparse.Namespace(**args_dict)
    model = T5FineTuner(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=args.output_dir + "/checkpoint.pth",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        accelerator="cpu",
        # gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision=32,
        gradient_clip_val=args.max_grad_norm,
        callbacks=[checkpoint_callback, LoggingCallback()],
    )

    def get_dataset(tokenizer, type_path, args):
        tokenizer.max_length = args.max_seq_length
        tokenizer.model_max_length = args.max_seq_length
        jnlpba = load_dataset("jnlpba", split=["train[:50]", "validation[:50]"])
        jnlpba = DatasetDict({"train": jnlpba[0], "validation": jnlpba[1]})
        dataset = jnlpba
        return JnlpbDataset(
            tokenizer=tokenizer, dataset=dataset, type_path=type_path, portion=0
        )

    # call get dataset to get the dataset

    trainer = pl.Trainer(**train_params)

    trainer.fit(model)

    wandb.alert(
        title="End of training.",
        text="Training finished successfully.",
    )

    wandb.finish()
