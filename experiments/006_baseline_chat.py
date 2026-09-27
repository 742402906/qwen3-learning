import torch
import json
from pathlib import Path
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================
# 1. 配置
# =====================================

MODEL_NAME = "Qwen/Qwen3-0.6B"

OUTPUT_DIR = Path("outputs/experiment_006_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = OUTPUT_DIR / "baseline_results.jsonl"

MAX_NEW_TOKENS = 512

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Device:", device)

# =====================================
# 2. 加载原始模型
# =====================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading Qwen3-0.6B...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.to(device)
model.eval()

print("Model loaded successfully!")
print("输入 exit 退出，输入 clear 清空当前对话。")

# =====================================
# 3. 对话历史
# =====================================

messages = []

# =====================================
# 4. 交互式聊天
# =====================================

while True:

    user_input = input("\n你：").strip()

    if not user_input:
        continue

    if user_input.lower() in ["exit", "quit"]:
        print("退出聊天。")
        break

    if user_input.lower() == "clear":
        messages = []
        print("对话历史已清空。")
        continue

    # 加入当前用户问题
    messages.append({
        "role": "user",
        "content": user_input
    })

    # 使用 Qwen3 自带的 Chat Template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    # =================================
    # 5. 模型推理
    # =================================

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 只提取新生成的 Token
    input_length = inputs["input_ids"].shape[1]

    new_tokens = output_ids[0][input_length:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    print("\nQwen3-0.6B：")
    print(answer)

    # =================================
    # 6. 保存训练前的对话
    # =================================

    record = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "device": device,
        "user": user_input,
        "assistant": answer
    }

    with open(
        RESULT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

    # 将回答加入上下文，支持多轮聊天
    messages.append({
        "role": "assistant",
        "content": answer
    })

    print(f"\n[已保存至 {RESULT_FILE}]")
