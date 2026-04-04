import math
import re
from typing import Optional


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

# 允许一些宽松写法
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


def _to_text(x) -> str:
    if x is None:
        return ""
    return str(x)


def _extract_last_tag(text: str, tag: str) -> str:
    text = _to_text(text)
    matches = re.findall(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return matches[-1].strip() if matches else ""


def _normalize_label(x) -> Optional[str]:
    """
    支持：
    1) int label: 0~5
    2) str label: FALSE / HALF_TRUE / Half True ...
    3) full completion / full solution text: 自动从 <answer> 中解析
    """
    if x is None:
        return None

    if isinstance(x, int):
        return ID2LABEL.get(int(x))

    s = _to_text(x).strip()
    if not s:
        return None

    # 若传入的是完整输出，优先从 <answer> 中提取
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

    # 兜底：在长文本里搜索标签短语
    for alias, canon in LABEL_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", s_up):
            return canon

    return None


def _resolve_gold_labels(
    gold_label=None,
    label=None,
    **kwargs,
):
    """
    优先级：
    gold_label > label > solution
    """
    src = None
    if gold_label is not None:
        src = gold_label
    elif label is not None:
        src = label
    else:
        raise ValueError(
            "Cannot find gold labels. Expected one of: gold_label, label, solution"
        )

    return [_normalize_label(x) for x in src]


def _tokenize_content_words(text: str) -> list[str]:
    text = _to_text(text).lower()
    tokens = re.findall(r"[a-z][a-z0-9\-']+|\d+(?:\.\d+)?", text)
    return [t for t in tokens if (len(t) >= 3 or t.isdigit()) and t not in STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    sents = re.split(r"[\n\r]+|(?<=[\.\!\?;])\s+", _to_text(text).strip())
    sents = [re.sub(r"\s+", " ", s).strip() for s in sents]
    return [s for s in sents if s]


def extract_prediction(content: str):
    explanation = _extract_last_tag(content, "explanation")
    answer = _extract_last_tag(content, "answer")
    pred_label = _normalize_label(answer if answer else content)
    return {
        "explanation": explanation,
        "answer": answer,
        "label": pred_label,
    }


def factcheck_format_reward(completions, **kwargs):
    """
    结构必须是:
    <explanation>...</explanation>
    <answer>...</answer>
    """
    pattern = r"^<explanation>\s*.*?\s*</explanation>\s*<answer>\s*.*?\s*</answer>\s*$"
    contents = [completion[0]["content"] for completion in completions]
    return [
        1.0 if re.match(pattern, c, flags=re.DOTALL | re.IGNORECASE) else 0.0
        for c in contents
    ]


def factcheck_tag_count_reward(completions, **kwargs):
    def count_tags(text: str) -> float:
        score = 0.0
        if text.count("<explanation>") == 1:
            score += 0.25
        if text.count("</explanation>") == 1:
            score += 0.25
        if text.count("<answer>") == 1:
            score += 0.25
        if text.count("</answer>") == 1:
            score += 0.25
        return score

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def factcheck_label_reward(completions, gold_label=None, label=None, **kwargs):
    """
    主奖励：exact label match
    exact = 2.0
    invalid label = -0.25
    valid but wrong = 0.0
    """
    contents = [completion[0]["content"] for completion in completions]
    golds = _resolve_gold_labels(gold_label=gold_label, label=label, **kwargs)

    rewards = []
    for content, gold in zip(contents, golds):
        pred = extract_prediction(content)["label"]
        if gold is None:
            rewards.append(0.0)
        elif pred is None:
            rewards.append(-0.25)
        elif pred == gold:
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards


def factcheck_ordinal_reward(completions, gold_label=None, label=None, **kwargs):
    """
    六分类有明显序关系时使用：
    PANTS_FIRE < FALSE < BARELY_TRUE < HALF_TRUE < MOSTLY_TRUE < TRUE

    exact 之外再给一个较弱的 ordinal shaping。
    exact 也会得到这部分奖励，因此这是“辅助项”。
    """
    contents = [completion[0]["content"] for completion in completions]
    golds = _resolve_gold_labels(gold_label=gold_label, label=label, **kwargs)

    rewards = []
    for content, gold in zip(contents, golds):
        pred = extract_prediction(content)["label"]
        if pred is None or gold is None:
            rewards.append(0.0)
            continue

        dist = abs(LABEL2ID[pred] - LABEL2ID[gold])
        # dist=0 -> 0.5, dist=5 -> 0.0
        reward = max(0.0, 1.0 - dist / 5.0) * 0.5
        rewards.append(float(reward))
    return rewards


def factcheck_explanation_quality_reward(completions, **kwargs):
    """
    只做弱约束，不拿 gold explanation 做硬匹配。
    目标：
    1) 有 explanation
    2) 解释长度适中
    3) 不是一句话反复重复
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content in contents:
        explanation = extract_prediction(content)["explanation"]
        if not explanation:
            rewards.append(0.0)
            continue

        tokens = _tokenize_content_words(explanation)
        sents = _split_sentences(explanation)

        score = 0.10  # 有 explanation 的基础分

        n_tok = len(tokens)
        if 20 <= n_tok <= 120:
            score += 0.20
        elif 8 <= n_tok < 20 or 120 < n_tok <= 220:
            score += 0.10

        if len(sents) >= 2:
            score += 0.05

        if sents:
            norm_sents = [re.sub(r"\s+", " ", s.lower()).strip() for s in sents if s.strip()]
            unique_ratio = len(set(norm_sents)) / max(1, len(norm_sents))
            if unique_ratio >= 0.80:
                score += 0.15
            elif unique_ratio >= 0.60:
                score += 0.08
            elif unique_ratio < 0.40:
                score -= 0.10

        rewards.append(float(max(0.0, min(0.50, score))))

    return rewards


def factcheck_grounding_reward(
    completions,
    evidences,
    claims,
    **kwargs,
):
    """
    evidence-grounding 弱约束：
    - 解释与 evidence 有一定内容词重叠
    - 若 evidence 中有数字/年份，解释命中可额外加分
    - 若解释几乎只在重复 claim 而未触及 evidence，轻微扣分

    注意：这是弱约束，不是严格事实核验器。
    """
    contents = [completion[0]["content"] for completion in completions]

    rewards = []
    for content, evs, cl in zip(contents, evidences, claims):
        explanation = extract_prediction(content)["explanation"]
        if not explanation:
            rewards.append(0.0)
            continue

        if evs is None:
            rewards.append(0.0)
            continue

        if isinstance(evs, list):
            ev_text = " ".join([_to_text(x) for x in evs])
        else:
            ev_text = _to_text(evs)

        explanation_set = set(_tokenize_content_words(explanation))
        ev_set = set(_tokenize_content_words(ev_text))
        claim_set = set(_tokenize_content_words(_to_text(cl)))

        if not explanation_set or not ev_set:
            rewards.append(0.0)
            continue

        overlap = explanation_set & ev_set
        lexical = min(0.35, 0.35 * len(overlap) / 6.0)

        ev_nums = set(re.findall(r"\d+(?:\.\d+)?", ev_text))
        explanation_nums = set(re.findall(r"\d+(?:\.\d+)?", explanation))
        num_bonus = 0.10 if ev_nums and (ev_nums & explanation_nums) else 0.0

        claim_overlap = len(explanation_set & claim_set)
        claim_only_penalty = 0.0
        if len(overlap) < 2 and claim_overlap >= 3:
            claim_only_penalty = 0.10

        score = lexical + num_bonus - claim_only_penalty
        rewards.append(float(max(0.0, min(0.50, score))))

    return rewards


def get_factcheck_cosine_scaled_reward(
    min_value_wrong: float = -0.5,
    max_value_wrong: float = 0.0,
    min_value_correct: float = 0.2,
    max_value_correct: float = 0.6,
    max_len: int = 1000,
):
    """
    参考 rewards_math.py 的 cosine length reward 思路，
    但 correctness 改成“标签是否预测正确”。
    """
    def cosine_scaled_reward(completions, solution=None, gold_label=None, label=None, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        golds = _resolve_gold_labels(solution=solution, gold_label=gold_label, label=label, **kwargs)

        rewards = []
        for content, gold in zip(contents, golds):
            pred = extract_prediction(content)["label"]
            is_correct = (pred is not None and gold is not None and pred == gold)

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


def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.5):
    if max_penalty > 0:
        raise ValueError("max_penalty should be <= 0")

    def zipngram(text: str, ngram_size: int):
        words = _to_text(text).lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
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