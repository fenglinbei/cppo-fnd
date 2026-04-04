import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, List as HFList, load_from_disk, load_dataset

# ====== 1. 按你的任务定义标签 ======
LABEL_NAMES = [
    "PANTS_FIRE",
    "FALSE",
    "BARELY_TRUE",
    "HALF_TRUE",
    "MOSTLY_TRUE",
    "TRUE",
]

LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}

FEATURES = Features({
    "id": Value("int64"),
    "claim": Value("string"),
    "label": ClassLabel(names=LABEL_NAMES),
    "explanation": Value("string"),
    "evidence": HFList(Value("string")),
})


# ====== 2. 读取单个 json 文件 ======
def read_json_file(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} 不是 list[dict] 格式，而是 {type(data)}")

    return data


# ====== 3. 规范化一条样本 ======
def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)

    # id
    out["id"] = int(out["id"])

    # claim / explanation
    out["claim"] = str(out["claim"])
    out["explanation"] = str(out["explanation"])

    # evidence: 保证是 list[str]
    evidence = out.get("evidence", [])
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list):
        raise ValueError(f"evidence 字段不是 list: {evidence}")
    out["evidence"] = [str(x) for x in evidence]

    # label:
    # 兼容三种情况：
    # 1) 已经是 int，如 0~5
    # 2) 是字符串标签，如 "FALSE"
    # 3) 是枚举序列化后的字符串形式
    label = out["label"]
    if isinstance(label, int):
        label_id = label
    elif isinstance(label, str):
        if label in LABEL2ID:
            label_id = LABEL2ID[label]
        else:
            raise ValueError(f"未知 label 字符串: {label}")
    else:
        raise ValueError(f"不支持的 label 类型: {type(label)} | 值={label}")

    if not (0 <= label_id < len(LABEL_NAMES)):
        raise ValueError(f"label 超出范围: {label_id}")

    out["label"] = label_id

    return out


# ====== 4. 单个 split 转成 HF Dataset ======
def build_split_dataset(path: str | Path) -> Dataset:
    raw_records = read_json_file(path)
    print(raw_records[0])
    records = [normalize_record(x) for x in raw_records]
    ds = Dataset.from_list(records, features=FEATURES)
    return ds


# ====== 5. 构造 DatasetDict 并保存 ======
def build_hf_dataset_dict(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
) -> DatasetDict:
    dataset_dict = DatasetDict({
        "train": build_split_dataset(train_path),
        "validation": build_split_dataset(val_path),  # 建议用 validation，不用 val
        "test": build_split_dataset(test_path),
    })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    return dataset_dict


if __name__ == "__main__":
    # ds = build_hf_dataset_dict(
    #     train_path="data/LIAR-RAW/train.json",
    #     val_path="data/LIAR-RAW/val.json",
    #     test_path="data/LIAR-RAW/test.json",
    #     output_dir="data/liar-raw",
    # )

    # print(ds)
    # print(ds["train"].features)
    # print(ds["train"][0])

    # # 重新加载测试
    # ds2 = load_from_disk("data/liar-raw")
    # print(ds2)
    # print(ds2["train"])

    dataset = load_dataset("data/liar-raw")
    print(dataset)

