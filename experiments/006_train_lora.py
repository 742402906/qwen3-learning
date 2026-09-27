import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


# ========================================
# 1. 配置
# ========================================

MODEL_NAME = "Qwen/Qwen3-0.6B"

DATA_PATH = Path("data/experiment_006_train.jsonl")

OUTPUT_DIR = Path("outputs/experiment_006_lora")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_LENGTH = 512

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ========================================
# 2. 加载训练数据
# ========================================

train_data = []

with open(DATA_PATH, "r", encoding="utf-8") as f:

    for line in f:
        if line.strip():
            train_data.append(json.loads(line))

print("Training examples:", len(train_data))


# ========================================
# 3. 加载 Tokenizer 和基座模型
# ========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.to(device)
model.config.use_cache = False


# ========================================
# 4. 配置 LoRA
# ========================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()


# ========================================
# 5. 将对话转换成训练样本
# ========================================

def encode_example(example):

    messages = example["messages"]

    # 完整对话，包含 Assistant 答案
    # transformers 5.x 的 apply_chat_template 返回 BatchEncoding，需取 ["input_ids"]
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False
    )["input_ids"]

    # 只有 User 部分，加上 Assistant 的生成起始标记
    prompt_messages = messages[:-1]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True
    )["input_ids"]

    # 截断到最大长度
    full_ids = full_ids[:MAX_LENGTH]

    prompt_length = min(
        len(prompt_ids),
        len(full_ids)
    )

    input_ids = torch.tensor(
        full_ids,
        dtype=torch.long
    )

    labels = input_ids.clone()

    # 不计算 User 和 Assistant Header 的 Loss
    labels[:prompt_length] = -100

    # 检查是否存在有效监督 Token
    if (labels != -100).sum().item() == 0:
        raise ValueError(
            "No assistant target tokens after truncation"
        )

    return {
        "input_ids": input_ids,
        "labels": labels
    }


encoded_data = [
    encode_example(example)
    for example in train_data
]

print("Encoded examples:", len(encoded_data))


# ========================================
# 6. 创建优化器
# ========================================

optimizer = torch.optim.AdamW(
    [
        p for p in model.parameters()
        if p.requires_grad
    ],
    lr=LEARNING_RATE
)

print("\nTrainable parameters ready.")


# ========================================
# 7. 开始训练
# ========================================

model.train()

loss_history = []

for epoch in range(EPOCHS):

    epoch_losses = []

    print(f"\n========== Epoch {epoch + 1} ==========")

    for step, sample in enumerate(encoded_data):

        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)

        attention_mask = torch.ones_like(input_ids)

        # 清空上一轮梯度
        optimizer.zero_grad(set_to_none=True)

        # 前向传播 + 计算交叉熵 Loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # 反向传播：计算 LoRA 参数的梯度
        loss.backward()

        # 梯度裁剪，降低梯度爆炸风险
        torch.nn.utils.clip_grad_norm_(
            [
                p for p in model.parameters()
                if p.requires_grad
            ],
            max_norm=1.0
        )

        # 更新 LoRA 参数
        optimizer.step()

        loss_value = loss.item()

        epoch_losses.append(loss_value)
        loss_history.append(loss_value)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"Step {step + 1}/{len(encoded_data)} "
            f"Loss: {loss_value:.4f}"
        )

    mean_loss = sum(epoch_losses) / len(epoch_losses)

    print(
        f"Epoch {epoch + 1} Mean Loss: "
        f"{mean_loss:.4f}"
    )


# ========================================
# 8. 保存 LoRA Adapter
# ========================================

model.save_pretrained(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nLoRA Adapter saved to:", OUTPUT_DIR)


# ========================================
# 9. 绘制 Loss 曲线
# ========================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(loss_history) + 1),
    loss_history,
    marker="o"
)

plt.xlabel("Training Step")
plt.ylabel("Training Loss")
plt.title("Experiment 006 - LoRA Training Loss")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "training_loss.png",
    dpi=200
)

plt.close()

print("Loss plot saved.")
print("Experiment 006 completed.")
