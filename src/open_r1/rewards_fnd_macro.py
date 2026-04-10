
import math
import re
from typing import Any, Optional


LABEL_NAMES = [
    "PANTS_FIRE",
    "FALSE",
    "BARELY_TRUE",
    "HALF_TRUE",
    "MOSTLY_TRUE",
    "TRUE",
]
LABEL2ID = {x: i for i, x in enumerate(LABEL_NAMES)}
ID2LABEL = {i: x for i, x in enumerate(LABEL_NAMES)}

# 训练集分布：LIAR-RAW train
DEFAULT_CLASS_COUNTS = [812, 1985, 1611, 2087, 1950, 1647]


LABEL_ALIASES = {
    "PANTS FIRE": "PANTS_FIRE",
    "PANTS_FIRE": "PANTS_FIRE",
    "FALSE": "FALSE",
    "BARELY TRUE": "BARELY_TRUE",
    "BARELY_TRUE": "BARELY_TRUE",
    "HALF TRUE": "HALF_TRUE",
    "HALF_TRUE": "HALF_TRUE",
    "MOSTLY TRUE": "MOSTLY_TRUE",
    "MOSTLY_TRUE": "MOSTLY_TRUE",
    "TRUE": "TRUE",
}


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than",
    "to", "of", "in", "on", "for", "from", "with", "without",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "it", "its", "they", "them", "their",
    "he", "she", "his", "her", "we", "our", "you", "your",
    "as", "at", "by", "about", "into", "over", "after", "before",
    "very", "more", "most", "less", "least",
    "not", "no", "yes",
}


# 预测某个类出错时，给一个很轻的 class-aware FP penalty。
# 这里用 inverse-sqrt frequency，均值归一化到 1。
def _build_pred_fp_weights(counts=DEFAULT_CLASS_COUNTS):
    inv_sqrt = [1.0 / math.sqrt(max(c, 1)) for c in counts]
    mean_v = sum(inv_sqrt) / len(inv_sqrt)
    return [x / mean_v for x in inv_sqrt]


PRED_FP_WEIGHTS = _build_pred_fp_weights()


DEFAULT_GUIDED_DECODING_REGEX = (
    r"(?s)^<explanation>.*?</explanation>\s*"
    r"<evidence_used>(?:none|[1-5](?:\s*,\s*[1-5]){0,4})</evidence_used>\s*"
    r"<answer>(?:PANTS_FIRE|FALSE|BARELY_TRUE|HALF_TRUE|MOSTLY_TRUE|TRUE)</answer>$"
)


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _completion_to_text(completion: Any) -> str:
    """
    同时兼容：
    1) non-conversational: str
    2) conversational: [{"role": "assistant", "content": "..."}]
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list) and len(completion) > 0:
        last = completion[-1]
        if isinstance(last, dict) and "content" in last:
            return _to_text(last["content"])

    if isinstance(completion, dict) and "content" in completion:
        return _to_text(completion["content"])

    return _to_text(completion)


def _extract_last_tag(text: str, tag: str) -> str:
    text = _to_text(text)
    matches = re.findall(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return matches[-1].strip() if matches else ""


def _normalize_label(x) -> Optional[str]:
    if x is None:
        return None

    if isinstance(x, int):
        return ID2LABEL.get(int(x))

    s = _to_text(x).strip()
    if not s:
        return None

    ans = _extract_last_tag(s, "answer")
    if ans:
        s = ans

    s_up = re.sub(r"[^A-Z]+", " ", s.upper()).strip()
    if not s_up:
        return None

    if s_up in LABEL_ALIASES:
        return LABEL_ALIASES[s_up]

    s_us = re.sub(r"\s+", "_", s_up)
    if s_us in LABEL2ID:
        return s_us

    for alias, canon in LABEL_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", s_up):
            return canon

    return None


def _resolve_gold_labels(gold_label=None, label=None, **kwargs):
    src = None
    if gold_label is not None:
        src = gold_label
    elif label is not None:
        src = label
    else:
        raise ValueError("Cannot find gold labels. Expected one of: gold_label, label")

    return [_normalize_label(x) for x in src]


def _resolve_batch_field(length: int, *candidate_values):
    """
    在 kwargs 中拿到 batch 对齐字段。
    若是单个值，则广播到 batch 大小。
    若本来就是 batch list，直接返回。
    """
    for value in candidate_values:
        if value is None:
            continue
        if isinstance(value, list) and len(value) == length:
            return value
        return [value] * length
    return [None] * length


def _tokenize_content_words(text: str) -> list[str]:
    text = _to_text(text).lower()
    tokens = re.findall(r"[a-z][a-z0-9\-']+|\d+(?:\.\d+)?", text)
    return [t for t in tokens if (len(t) >= 3 or t.isdigit()) and t not in STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    sents = re.split(r"[\n\r]+|(?<=[\.\!\?;])\s+", _to_text(text).strip())
    sents = [re.sub(r"\s+", " ", s).strip() for s in sents]
    return [s for s in sents if s]


def _parse_evidence_ids(text: str) -> list[int]:
    text = _to_text(text).strip().lower()
    if not text or text == "none":
        return []
    ids = []
    for x in re.findall(r"\d+", text):
        try:
            ids.append(int(x))
        except ValueError:
            pass
    # 去重但保持顺序
    dedup = []
    seen = set()
    for x in ids:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


def _flatten_evidence_item(item: Any) -> list[str]:
    """
    兼容：
    - list[str]
    - list[dict(text=...)] / list[dict(content=...)]
    - dict(text=...) / dict(content=...)
    - None
    """
    if item is None:
        return []

    if isinstance(item, dict):
        if "text" in item:
            return [_to_text(item["text"])]
        if "content" in item:
            return [_to_text(item["content"])]
        return [_to_text(item)]

    if isinstance(item, list):
        out = []
        for x in item:
            out.extend(_flatten_evidence_item(x))
        return [t for t in out if t.strip()]

    text = _to_text(item).strip()
    return [text] if text else []

def extract_prediction(content: str):
    explanation = _extract_last_tag(content, "explanation")
    evidence_used_raw = _extract_last_tag(content, "evidence_used")
    answer = _extract_last_tag(content, "answer")
    pred_label = _normalize_label(answer if answer else content)

    evidence_ids = []
    raw = evidence_used_raw.strip().lower()
    if raw and raw != "none":
        evidence_ids = [int(x) for x in re.findall(r"\d+", evidence_used_raw)]
        evidence_ids = sorted(set(evidence_ids))

    return {
        "explanation": explanation,
        "evidence_used": evidence_used_raw,
        "evidence_ids": evidence_ids,
        "answer": answer,
        "label": pred_label,
    }


def _is_correct_label(pred_label: Optional[str], gold_label: Optional[str]) -> bool:
    return pred_label is not None and gold_label is not None and pred_label == gold_label


FACTCHECK_FORMAT_PATTERN = (
    r"^<explanation>\s*.*?\s*</explanation>\s*"
    r"<evidence_used>\s*(?:none|[1-5](?:\s*,\s*[1-5]){0,4})\s*</evidence_used>\s*"
    r"<answer>\s*(?:PANTS_FIRE|FALSE|BARELY_TRUE|HALF_TRUE|MOSTLY_TRUE|TRUE)\s*</answer>\s*$"
)

def factcheck_format_reward(completions, **kwargs):
    contents = [
        c[0]["content"] if isinstance(c, list) else str(c)
        for c in completions
    ]
    return [
        1.0 if re.match(FACTCHECK_FORMAT_PATTERN, x, flags=re.DOTALL | re.IGNORECASE) else 0.0
        for x in contents
    ]

def factcheck_tag_count_reward(completions, **kwargs):
    contents = [_completion_to_text(c) for c in completions]

    def count_tags(text: str) -> float:
        score = 0.0
        if text.count("<explanation>") == 1:
            score += 0.20
        if text.count("</explanation>") == 1:
            score += 0.20
        if text.count("<evidence_used>") == 1:
            score += 0.20
        if text.count("</evidence_used>") == 1:
            score += 0.20
        if text.count("<answer>") == 1:
            score += 0.10
        if text.count("</answer>") == 1:
            score += 0.10
        return score

    return [count_tags(c) for c in contents]


def factcheck_label_reward(
    completions,
    gold_label=None,
    pred_fp_weights=None,
    **kwargs,
):
    """
    主奖励：
    1) exact match 强正奖励
    2) parse failure 强负奖励
    3) wrong label 给负奖励，并根据 ordinal distance + 预测类 rarity 做细分

    这样在“同一 claim 的多个候选都错了”时，GRPO 仍然能区分：
    - 接近 gold 的错，比很远的错更好
    - 随便乱猜稀有类，处罚更大一点
    """
    contents = [_completion_to_text(c) for c in completions]
    golds = _resolve_gold_labels(gold_label=gold_label, **kwargs)
    pred_fp_weights = pred_fp_weights or PRED_FP_WEIGHTS

    rewards = []
    for content, gold in zip(contents, golds):
        pred = extract_prediction(content)["label"]

        if gold is None:
            rewards.append(0.0)
            continue

        if pred is None:
            rewards.append(-1.00)
            continue

        if pred == gold:
            rewards.append(2.25)
            continue

        dist = abs(LABEL2ID[pred] - LABEL2ID[gold])  # 1~5
        fp_pen = float(pred_fp_weights[LABEL2ID[pred]])  # rare predicted class => slightly larger
        wrong_reward = -0.15 - 0.12 * dist - 0.10 * fp_pen
        rewards.append(float(wrong_reward))

    return rewards


def factcheck_ordinal_reward(completions, gold_label=None, **kwargs):
    """
    可选的小辅助项。
    这里只给很弱的 shaping；推荐和主 label reward 一起用时，把外部权重设得很小。
    """
    contents = [_completion_to_text(c) for c in completions]
    golds = _resolve_gold_labels(gold_label=gold_label, **kwargs)

    rewards = []
    for content, gold in zip(contents, golds):
        pred = extract_prediction(content)["label"]
        if pred is None or gold is None:
            rewards.append(0.0)
            continue
        dist = abs(LABEL2ID[pred] - LABEL2ID[gold])
        rewards.append(float(max(0.0, 1.0 - dist / 5.0) * 0.25))
    return rewards


def factcheck_evidence_usage_reward(
    completions,
    evidence=None,
    retrieved_evidence=None,
    reports=None,
    gold_label=None,
    gate_on_correct: bool = False,
    **kwargs,
):
    """
    没有 gold evidence id 时，先奖励“合法使用 evidence ids”：
    - evidence 存在时，最好选 1~3 个合法 id
    - 不能引用越界 id
    - 没有 evidence 时，使用 none 最好
    """
    contents = [_completion_to_text(c) for c in completions]
    evidences = _resolve_batch_field(len(contents), evidence, retrieved_evidence, reports)
    golds = _resolve_gold_labels(gold_label=gold_label, **kwargs) if gold_label is not None else [None] * len(contents)

    rewards = []
    for content, ev_item, gold in zip(contents, evidences, golds):
        pred = extract_prediction(content)
        pred_label = pred["label"]
        evidence_ids = pred["evidence_ids"]
        raw = pred["evidence_used"].strip().lower()
        ev_list = _flatten_evidence_item(ev_item)
        n_ev = len(ev_list)

        score = 0.0

        if n_ev == 0:
            if raw in {"", "none"}:
                score = 0.15
            elif evidence_ids:
                score = -0.10
            rewards.append(float(score))
            continue

        if raw == "":
            # 没写 evidence_used，不直接判死，但不给分
            score = 0.0
        elif raw == "none":
            score = -0.05
        else:
            valid_ids = [i for i in evidence_ids if 1 <= i <= n_ev]
            invalid_ids = [i for i in evidence_ids if not (1 <= i <= n_ev)]

            if len(valid_ids) == 0:
                score -= 0.10
            else:
                # 1~3 条最合适，超过 3 条不再继续加
                score += 0.06 + 0.05 * min(len(valid_ids), 3)

            if invalid_ids:
                score -= 0.12 * len(invalid_ids)

        if gate_on_correct and not _is_correct_label(pred_label, gold):
            score = min(score, 0.0)

        rewards.append(float(max(-0.30, min(0.30, score))))

    return rewards


def factcheck_explanation_quality_reward(
    completions,
    gold_label=None,
    gate_on_correct: bool = True,
    **kwargs,
):
    """
    解释质量弱约束：
    - 有 explanation
    - 长度适中
    - 不是简单重复

    默认 gate_on_correct=True：
    label 错时不给 explanation 正向加分。
    """
    contents = [_completion_to_text(c) for c in completions]
    golds = _resolve_gold_labels(gold_label=gold_label, **kwargs) if gold_label is not None else [None] * len(contents)

    rewards = []
    for content, gold in zip(contents, golds):
        pred = extract_prediction(content)
        pred_label = pred["label"]
        explanation = pred["explanation"]

        if not explanation:
            rewards.append(0.0)
            continue

        tokens = _tokenize_content_words(explanation)
        sents = _split_sentences(explanation)

        score = 0.05

        n_tok = len(tokens)
        if 20 <= n_tok <= 120:
            score += 0.15
        elif 8 <= n_tok < 20 or 120 < n_tok <= 220:
            score += 0.05
        else:
            score -= 0.03

        if len(sents) >= 2:
            score += 0.05

        if sents:
            norm_sents = [re.sub(r"\s+", " ", s.lower()).strip() for s in sents if s.strip()]
            unique_ratio = len(set(norm_sents)) / max(1, len(norm_sents))
            if unique_ratio >= 0.80:
                score += 0.10
            elif unique_ratio >= 0.60:
                score += 0.05
            elif unique_ratio < 0.40:
                score -= 0.08

        if gate_on_correct and gold is not None and not _is_correct_label(pred_label, gold):
            score = min(score, 0.0)

        rewards.append(float(max(-0.10, min(0.35, score))))

    return rewards


def factcheck_grounding_reward(
    completions,
    claim=None,
    evidence=None,
    retrieved_evidence=None,
    reports=None,
    gold_label=None,
    gate_on_correct: bool = True,
    **kwargs,
):
    """
    无 gold evidence id 时的 grounding 弱监督：
    - 若模型选了 evidence_used，则只看被选中的 evidence
    - 否则退化到看所有 top-k evidence
    - explanation 和 evidence 的内容词 / 数字有重合则加分
    - 只复述 claim、却不碰 evidence 时扣分
    - evidence id 越界时扣分

    默认也 gate_on_correct=True，避免 label 错但 explanation“写得像真的”时抬高总 reward。
    """
    contents = [_completion_to_text(c) for c in completions]
    claims = _resolve_batch_field(len(contents), claim)
    evidences = _resolve_batch_field(len(contents), evidence, retrieved_evidence, reports)
    golds = _resolve_gold_labels(gold_label=gold_label, **kwargs) if gold_label is not None else [None] * len(contents)

    rewards = []
    for content, cl, ev_item, gold in zip(contents, claims, evidences, golds):
        pred = extract_prediction(content)
        pred_label = pred["label"]
        explanation = pred["explanation"]
        evidence_ids = pred["evidence_ids"]

        if not explanation:
            rewards.append(0.0)
            continue

        ev_list = _flatten_evidence_item(ev_item)
        if len(ev_list) == 0:
            rewards.append(0.0)
            continue

        invalid_ids = [i for i in evidence_ids if not (1 <= i <= len(ev_list))]
        valid_ids = [i for i in evidence_ids if 1 <= i <= len(ev_list)]

        if valid_ids:
            chosen_evs = [ev_list[i - 1] for i in valid_ids]
        else:
            chosen_evs = ev_list

        ev_text = " ".join(chosen_evs)
        explanation_set = set(_tokenize_content_words(explanation))
        ev_set = set(_tokenize_content_words(ev_text))
        claim_set = set(_tokenize_content_words(_to_text(cl)))

        if not explanation_set or not ev_set:
            rewards.append(0.0)
            continue

        overlap = explanation_set & ev_set
        lexical = min(0.26, 0.26 * len(overlap) / 6.0)

        ev_nums = set(re.findall(r"\d+(?:\.\d+)?", ev_text))
        explanation_nums = set(re.findall(r"\d+(?:\.\d+)?", explanation))
        num_bonus = 0.08 if ev_nums and (ev_nums & explanation_nums) else 0.0

        claim_overlap = len(explanation_set & claim_set)
        claim_only_penalty = 0.0
        if len(overlap) < 2 and claim_overlap >= 3:
            claim_only_penalty = 0.10

        selection_bonus = 0.08 if valid_ids else 0.0
        invalid_penalty = 0.10 * len(invalid_ids)

        score = selection_bonus + lexical + num_bonus - claim_only_penalty - invalid_penalty

        if gate_on_correct and gold is not None and not _is_correct_label(pred_label, gold):
            score = min(score, 0.0)

        rewards.append(float(max(-0.30, min(0.45, score))))

    return rewards


def get_factcheck_cosine_scaled_reward(
    min_value_wrong: float = -0.50,
    max_value_wrong: float = 0.0,
    min_value_correct: float = 0.20,
    max_value_correct: float = 0.60,
    max_len: int = 800,
):
    """
    正确答案更偏好短而有用的输出，错误答案更偏好更短。
    这个项只建议给很小的外部权重。
    """
    def cosine_scaled_reward(completions, gold_label=None, label=None, **kwargs):
        contents = [_completion_to_text(c) for c in completions]
        golds = _resolve_gold_labels(gold_label=gold_label, label=label, **kwargs)

        rewards = []
        for content, gold in zip(contents, golds):
            pred = extract_prediction(content)["label"]
            is_correct = _is_correct_label(pred, gold)

            gen_len = min(len(content), max_len)
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                min_value = min_value_wrong
                max_value = max_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.30):
    if max_penalty > 0:
        raise ValueError("max_penalty should be <= 0")

    def zipngram(text: str, ngram_size: int):
        words = _to_text(text).lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs):
        contents = [_completion_to_text(c) for c in completions]
        rewards = []

        for text in contents:
            if not text:
                rewards.append(0.0)
                continue

            words = text.split()
            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(text, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1.0 - len(ngrams) / max(1, total)
            rewards.append(float(scaling * max_penalty))

        return rewards

    return repetition_penalty_reward
