from datasets import load_dataset, load_metric
from datasets import DatasetDict


jnlpba = load_dataset("jnlpba", split=["train", "validation"])

jnlpba = DatasetDict({"train": jnlpba[0], "validation": jnlpba[1]})


class DatasetAnalyzer:
    def __init__(self, dataset_dict):
        self.dataset_dict = dataset_dict
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
                unique_attributes.add(self.mapping[attribute])
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



analyzer = DatasetAnalyzer(jnlpba)

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
