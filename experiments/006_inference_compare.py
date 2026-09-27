import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "outputs/experiment_006_lora"

BASELINE_FILE = Path("outputs/experiment_006_baseline/baseline_results.jsonl")
RESULT_FILE = Path("outputs/experiment_006_lora/tuned_results.jsonl")

# 与 baseline 实验保持一致（MAX_NEW_TOKENS=512），保证对比公平
MAX_NEW_TOKENS = 512

device = "mps" if torch.backends.mps.is_available() else "cpu"

# 与 baseline 相同的 5 个问题
QUESTIONS = [
    "针对智能摄像头的陌生人徘徊告警，生成一条测试用例。",
    "针对陌生人进入禁区，生成一条 JSON 格式的测试用例，包含 scenario、precondition、action、expected_result、test_type。",
    "请设计一个测试场景，验证摄像头是否会把已登记的熟人误报为陌生人。",
    "一个人进入监控区域后停留了8秒，系统没有触发告警。请分析可能的原因，并给出排查步骤。",
    "请解释精确率和召回率的区别，并分别给出智能摄像头人脸识别的计算公式。",
]

print("Device:", device)

# =====================================
# 1. 加载 Tokenizer 和基座模型
# =====================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
).to(device)

# =====================================
# 2. 加载 LoRA Adapter
# =====================================

print("Loading LoRA adapter from", ADAPTER_PATH, "...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()
print("Adapter loaded successfully!")

# =====================================
# 3. 逐题推理（每题独立对话，等价于 baseline 的 clear）
# =====================================

for question in QUESTIONS:

    messages = [
        {"role": "user", "content": question}
    ]

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

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    print("\n问题：", question)
    print("\nQwen3-0.6B + LoRA：")
    print(answer)

    record = {
        "timestamp": datetime.now().isoformat(),
        "model": BASE_MODEL,
        "adapter": ADAPTER_PATH,
        "device": device,
        "user": question,
        "assistant": answer
    }

    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\n[全部完成，已保存至", RESULT_FILE, "]")
