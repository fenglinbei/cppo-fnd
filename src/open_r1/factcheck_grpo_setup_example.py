
import torch
from trl.trainer.grpo_config import GRPOConfig

from grpo_trainer_fnd_macro import GRPOTrainer
from rewards_fnd_macro import (
    DEFAULT_GUIDED_DECODING_REGEX,
    factcheck_format_reward,
    factcheck_label_reward,
    factcheck_evidence_usage_reward,
    factcheck_grounding_reward,
    factcheck_explanation_quality_reward,
    get_repetition_penalty_reward,
)

# 你的 train / eval dataset 至少需要这些列：
# prompt, gold_label, claim, evidence
# 其中 evidence 推荐是 list[str]，长度 <= 5；如果没有 evidence 就传 [].

args = GRPOConfig(
    output_dir="outputs/fnd-grpo",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_generations=8,
    max_prompt_length=2048,
    max_completion_length=192,
    num_iterations=1,
    learning_rate=1e-6,
    beta=0.02,
    epsilon=0.2,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    remove_unused_columns=False,  # 最稳妥；trainer 里也做了兜底
    use_vllm=True,
    temperature=0.9,
    # 强烈建议打开 guided decoding，保证 parse_rate
    vllm_guided_decoding_regex=DEFAULT_GUIDED_DECODING_REGEX,
)

# 这两个属性不是 GRPOConfig 原生字段，但可以在实例创建后挂上去。
# trainer 会读取它们，用在“group-normalization 之后”的 class balancing。
args.factcheck_train_class_counts = [812, 1985, 1611, 2087, 1950, 1647]
args.factcheck_class_balance_alpha = 0.35

reward_funcs = [
    factcheck_format_reward,
    factcheck_label_reward,                # 主奖励
    factcheck_evidence_usage_reward,       # 小辅助
    factcheck_grounding_reward,            # 小辅助，默认 gate_on_correct=True
    factcheck_explanation_quality_reward,  # 小辅助，默认 gate_on_correct=True
    get_repetition_penalty_reward(3, -0.20),
]

# 建议先从这组权重开始
reward_weights = [
    0.08,   # format
    1.00,   # label (dominant)
    0.12,   # evidence usage
    0.20,   # grounding
    0.10,   # explanation quality
    0.05,   # repetition penalty
]
args.reward_weights = reward_weights

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-3B-Instruct",
    reward_funcs=reward_funcs,
    args=args,
    train_dataset=train_dataset,
    eval_dataset={"base": eval_dataset},
    processing_class=tokenizer,
)

trainer.train()
