from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, List as HFList, load_from_disk, load_dataset

dataset = load_from_disk("data/liar-raw")

print(dataset["train"].features["label"])
count_dict = {label: 0 for label in dataset["train"].features["label"].names}

ID2LABEL = {
    0: "PANTS_FIRE",
    1: "FALSE",
    2: "BARELY_TRUE",
    3: "HALF_TRUE",
    4: "MOSTLY_TRUE",
    5: "TRUE",
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}

for item in dataset["train"]:
    count_dict[ID2LABEL[item["label"]]] += 1

print(count_dict)