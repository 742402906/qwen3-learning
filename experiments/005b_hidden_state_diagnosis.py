import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-0.6B"
TEXT = "什么是AI测试"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
).to(device)

model.eval()

inputs = tokenizer(
    TEXT,
    return_tensors="pt"
).to(device)

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0].tolist()
)

# 转成可读字符串用于打印
labels = [
    tokenizer.convert_tokens_to_string([t])
    for t in tokens
]

print("Tokens:", labels)
print("Model layers:", model.config.num_hidden_layers)
print("Hidden size:", model.config.hidden_size)

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

hs = outputs.hidden_states

print("\n===== Hidden State 逐层统计 =====")

for i, h in enumerate(hs):

    # [batch, seq, hidden]
    x = h[0].detach().float().cpu()

    print(f"\nHidden State {i}")
    print("Shape:", tuple(x.shape))

    for j, token in enumerate(labels):

        v = x[j]

        print(
            f"Token {j} ({token}): "
            f"L2={torch.linalg.vector_norm(v).item():.4f}, "
            f"RMS={torch.sqrt(torch.mean(v**2)).item():.4f}, "
            f"MaxAbs={v.abs().max().item():.4f}, "
            f"Finite={torch.isfinite(v).all().item()}"
        )

# 检查模型参数中是否存在非有限值
print("\n===== 模型参数检查 =====")

bad_params = []

for name, param in model.named_parameters():

    if not torch.isfinite(param).all().item():
        bad_params.append(name)

print("Non-finite parameter count:", len(bad_params))

if bad_params:
    print("Examples:", bad_params[:10])

# 检查最后一层输出与最终 RMSNorm 的关系
print("\n===== 最终归一化检查 =====")

last_before_norm = hs[-2][0].detach().float().cpu()
last_hidden = hs[-1][0].detach().float().cpu()

print("hs[-2] shape:", tuple(last_before_norm.shape))
print("hs[-1] shape:", tuple(last_hidden.shape))

# Qwen3 通常使用 model.norm 作为最终 RMSNorm
if hasattr(model.model, "norm"):

    with torch.no_grad():

        manual_final = model.model.norm(
            hs[-2]
        )[0].detach().float().cpu()

    diff = torch.max(
        torch.abs(manual_final - last_hidden)
    ).item()

    print("Max abs difference after final norm:", diff)

else:
    print("Final norm module not found at model.model.norm")

print("\nDiagnosis completed.")
