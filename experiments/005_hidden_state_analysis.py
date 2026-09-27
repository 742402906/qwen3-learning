import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# macOS中文字体，避免图中中文显示为方框
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Arial Unicode MS",
    "Hiragino Sans GB",
    "Heiti TC",
]
plt.rcParams["axes.unicode_minus"] = False

# ==========================
# 1. 基础配置
# ==========================

MODEL_NAME = "Qwen/Qwen3-0.6B"
TEXT = "什么是AI测试"

OUTPUT_DIR = Path("outputs/experiment_005")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

# ==========================
# 2. 加载模型
# ==========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.to(device)
model.eval()

# ==========================
# 3. Tokenize
# ==========================

inputs = tokenizer(
    TEXT,
    return_tensors="pt"
).to(device)

input_ids = inputs["input_ids"]

tokens = tokenizer.convert_ids_to_tokens(
    input_ids[0].tolist()
)

# 转成可读字符串，用于图表标签和打印
labels = [
    tokenizer.convert_tokens_to_string([t])
    for t in tokens
]

print("\nInput text:", TEXT)
print("Input IDs:", input_ids[0].tolist())
print("Tokens:", labels)

# ==========================
# 4. 提取所有层的 Hidden States
# ==========================

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

hidden_states = outputs.hidden_states

print("\nHidden states数量:", len(hidden_states))

for i, h in enumerate(hidden_states):
    print(
        f"Hidden State {i}:",
        tuple(h.shape)
    )

# ==========================
# 5. 转换为 NumPy
# ==========================

# 每个元素的形状:
# [batch_size, sequence_length, hidden_size]

# hidden_states[0] 是 Embedding 输出
# hidden_states[1] 是 Transformer Layer 0 输出
# hidden_states[2] 是 Transformer Layer 1 输出
# ...
# hidden_states[-1] 是最后一层输出

hs = [
    h[0].detach().float().cpu().numpy()
    for h in hidden_states
]

# hs[layer_index] 的形状:
# [sequence_length, hidden_size]

num_layers = len(hs) - 1
seq_len = len(tokens)

print("\nTransformer layers:", num_layers)
print("Sequence length:", seq_len)
print("Hidden size:", hs[0].shape[-1])

# ==========================
# 6. 计算每层的 Hidden State Norm
# ==========================

norm_records = []

for layer_idx, layer_h in enumerate(hs):

    # layer_h: [seq_len, hidden_size]

    norms = np.linalg.norm(
        layer_h,
        axis=-1
    )

    for token_idx, token in enumerate(labels):
        norm_records.append({
            "layer": layer_idx,
            "token_index": token_idx,
            "token": token,
            "hidden_state_norm": float(
                norms[token_idx]
            )
        })

norm_df = pd.DataFrame(norm_records)

norm_df.to_csv(
    OUTPUT_DIR / "hidden_state_norm.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================
# 7. 计算相邻层变化幅度与余弦相似度
# ==========================

change_records = []

for layer_idx in range(1, len(hs)):

    prev_h = hs[layer_idx - 1]
    curr_h = hs[layer_idx]

    for token_idx, token in enumerate(labels):

        v_prev = prev_h[token_idx]
        v_curr = curr_h[token_idx]

        # L2变化幅度
        delta = np.linalg.norm(
            v_curr - v_prev
        )

        # 相对变化幅度
        prev_norm = np.linalg.norm(v_prev)

        relative_delta = (
            delta / (prev_norm + 1e-12)
        )

        # 余弦相似度
        denom = (
            np.linalg.norm(v_prev)
            * np.linalg.norm(v_curr)
            + 1e-12
        )

        cosine_sim = np.dot(
            v_prev,
            v_curr
        ) / denom

        change_records.append({
            "layer": layer_idx,
            "token_index": token_idx,
            "token": token,
            "delta_l2": float(delta),
            "relative_delta": float(relative_delta),
            "cosine_similarity": float(cosine_sim)
        })

change_df = pd.DataFrame(change_records)

change_df.to_csv(
    OUTPUT_DIR / "hidden_state_changes.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================
# 8. 图一：Hidden State Norm
# ==========================

plt.figure(figsize=(12, 6))

for token_idx, token in enumerate(labels):

    sub = norm_df[
        norm_df["token_index"] == token_idx
    ]

    plt.plot(
        sub["layer"],
        sub["hidden_state_norm"],
        marker="o",
        markersize=3,
        label=f"Token {token_idx}: {token}"
    )

plt.xlabel("Layer index (0 = Embedding)")
plt.ylabel("Hidden State L2 Norm")
plt.title("Hidden State Norm Across Layers")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "hidden_state_norm.png",
    dpi=200
)

plt.close()

# ==========================
# 9. 图二：相邻层变化幅度
# ==========================

plt.figure(figsize=(12, 6))

for token_idx, token in enumerate(labels):

    sub = change_df[
        change_df["token_index"] == token_idx
    ]

    plt.plot(
        sub["layer"],
        sub["relative_delta"],
        marker="o",
        markersize=3,
        label=f"Token {token_idx}: {token}"
    )

plt.xlabel("Transformer Layer")
plt.ylabel("Relative Hidden State Change")
plt.title("Relative Hidden State Change Between Layers")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "hidden_state_change.png",
    dpi=200
)

plt.close()

# ==========================
# 10. 打印关键结果
# ==========================

print("\n===== 每个 Token 的最大相对变化层 =====")

for token_idx, token in enumerate(labels):

    sub = change_df[
        change_df["token_index"] == token_idx
    ]

    max_row = sub.loc[
        sub["relative_delta"].idxmax()
    ]

    print(
        f"Token {token_idx} ({token}): "
        f"Layer {int(max_row['layer'])}, "
        f"Relative change = "
        f"{max_row['relative_delta']:.4f}, "
        f"Cosine similarity = "
        f"{max_row['cosine_similarity']:.4f}"
    )

print("\n===== 输出文件 =====")

for path in OUTPUT_DIR.iterdir():
    print(path)

print("\nExperiment 005-A completed.")
