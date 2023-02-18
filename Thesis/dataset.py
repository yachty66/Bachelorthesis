from datasets import load_dataset, load_metric
from datasets import DatasetDict
import pandas as pd
import random
from datasets import load_dataset, load_metric
from datasets import DatasetDict, Dataset

jnlpba = load_dataset("jnlpba", split=["train[:18500]", "validation[:3500]"])

jnlpba = DatasetDict({"train": jnlpba[0], "validation": jnlpba[1]})


class JnlpbDataset:
    def __init__(self, dataset, portion, type_path):
        self.dataset = dataset[type_path]
        self.portion = portion
        self.remove()
        self.merge()
        self.apply()
        
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
        
        #print(merged_tags)
        #print(merged_tokens)
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
jnlpba = DatasetDict({"train": dataset_train, "validation": dataset_validation})

class DatasetAnalyzer:
    def __init__(self, dataset_dict, condition):
        self.dataset_dict = dataset_dict
        self.condition = condition
        # https://huggingface.co/datasets/jnlpba/raw/main/dataset_infos.json
        self.mapping = {
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
        self.mapping_new = {
            0: "O",
            1: "RNA",
            2: "DNA",
            3: "Cell_line",
            4: "Cell_type",
            5: "Protein",
        }
        
    def get_number_of_lines(self, split):
        return len(self.dataset_dict[split])
    
    def get_number_of_occurence(self, split):
        occurence = {}
        for example in self.dataset_dict[split]:
            for attribute in example["ner_tags"]:
                if attribute in occurence:
                    occurence[attribute] += 1
                else:
                    occurence[attribute] = 1
        return occurence

    def get_unique_attributes(self, split):
        unique_attributes = set()
        for example in self.dataset_dict[split]:
            for attribute in example["ner_tags"]:
                if self.condition:
                    unique_attributes.add(self.mapping[attribute])
                else:
                    unique_attributes.add(self.mapping_new[attribute])
        return unique_attributes

    def get_average_word_per_line(self, split):
        total_word_count = 0
        total_examples = 0
        for example in self.dataset_dict[split]:
            total_word_count += len(example["tokens"])
            total_examples += 1
        return total_word_count / total_examples

    def get_average_attribute_count(self, split):
        total_attribute_count = 0
        total_examples = 0
        for example in self.dataset_dict[split]:
            for attribute in example["ner_tags"]:
                if attribute != 0:
                    total_attribute_count += 1
            total_examples += 1
        return total_attribute_count / total_examples

analyzer = DatasetAnalyzer(jnlpba, condition=False)

#add number of lines
print("Number of lines in train split:", analyzer.get_number_of_lines("train"))
print("Number of lines in validation split:", analyzer.get_number_of_lines("validation"))
#add number of occurence for each attribute
print("Number of occurence for each attribute in train split:", analyzer.get_number_of_occurence("train"))
print("Number of occurence for each attribute in validation split:", analyzer.get_number_of_occurence("validation"))

print("Unique attributes train:", analyzer.get_unique_attributes("train"))
print("Unique attributes validation:", analyzer.get_unique_attributes("validation"))
print(
    "Average word length in train split:", analyzer.get_average_word_per_line("train")
)
print(
    "Average word length in validation split:",
    analyzer.get_average_word_per_line("validation"),
)
print(
    "Average attribute count in train split:",
    analyzer.get_average_attribute_count("train"),
)
print(
    "Average attribute count in validation split:",
    analyzer.get_average_attribute_count("validation"),
)

# seq2seq
# {'id': '1', 'tokens': ['Number', 'of', 'glucocorticoid receptors', 'in', 'lymphocytes', 'and', 'their', 'sensitivity', 'to', 'hormone', 'action', '.'], 'ner_tags': [0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0], '__index_level_0__': 0, 'spans': ['Protein: glucocorticoid receptors', 'Cell_type: lymphocytes']}

# tokenclassification
# {'id': '1', 'tokens': ['IL-2 gene', 'expression', 'and', 'NF-kappa B', 'activation', 'through', 'CD28', 'requires', 'reactive', 'oxygen', 'production', 'by', '5-lipoxygenase', '.'], 'ner_tags': [0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0], '__index_level_0__': 0}