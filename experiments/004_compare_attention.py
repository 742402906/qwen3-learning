import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

# macOS中文字体，避免图中中文显示为方框
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Arial Unicode MS",
    "Hiragino Sans GB",
    "Heiti TC",
]
plt.rcParams["axes.unicode_minus"] = False

MODEL_NAME = "Qwen/Qwen3-0.6B"
TEXT = "什么是AI测试"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    attn_implementation="eager"
)

model.to(device)
model.eval()

inputs = tokenizer(TEXT, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0].tolist()
)

# 转成可读字符串，用作图中的坐标标签
labels = [
    tokenizer.convert_tokens_to_string([t])
    for t in tokens
]

print("Tokens:")
for i, token in enumerate(tokens):
    print(i, repr(token))

with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True,
        use_cache=False,
        return_dict=True
    )

attentions = outputs.attentions

num_layers = len(attentions)
num_heads = attentions[0].shape[1]

print("\n层数:", num_layers)
print("每层Attention Head数:", num_heads)
print("每层Attention张量形状:", attentions[0].shape)

# ---------------------------------
# 实验A：比较同一层的不同Head
# ---------------------------------

layer_idx = 0

# 选择前4个Head；如果模型Head数量不足4则全部选择
head_indices = list(range(min(4, num_heads)))

fig, axes = plt.subplots(
    1,
    len(head_indices),
    figsize=(5 * len(head_indices), 4)
)

if len(head_indices) == 1:
    axes = [axes]

for ax, head_idx in zip(axes, head_indices):

    matrix = attentions[layer_idx][
        0, head_idx
    ].detach().float().cpu().numpy()

    im = ax.imshow(
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    ax.set_title(
        f"Layer {layer_idx}, Head {head_idx}"
    )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(labels)

    ax.set_xlabel("Key")
    ax.set_ylabel("Query")

fig.colorbar(im, ax=axes, label="Attention Weight")
plt.tight_layout()

plt.savefig(
    "outputs/004_same_layer_heads.png",
    dpi=200,
    bbox_inches="tight"
)

print("已保存: outputs/004_same_layer_heads.png")

plt.show()

# ---------------------------------
# 实验B：比较不同层的同一个Head
# ---------------------------------

head_idx = 0

# 选取浅层、中间层、最后一层
layer_indices = sorted(set([
    0,
    num_layers // 2,
    num_layers - 1
]))

fig, axes = plt.subplots(
    1,
    len(layer_indices),
    figsize=(5 * len(layer_indices), 4)
)

if len(layer_indices) == 1:
    axes = [axes]

for ax, layer_idx in zip(axes, layer_indices):

    matrix = attentions[layer_idx][
        0, head_idx
    ].detach().float().cpu().numpy()

    im = ax.imshow(
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    ax.set_title(
        f"Layer {layer_idx}, Head {head_idx}"
    )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(labels)

    ax.set_xlabel("Key")
    ax.set_ylabel("Query")

fig.colorbar(im, ax=axes, label="Attention Weight")
plt.tight_layout()

plt.savefig(
    "outputs/004_different_layers.png",
    dpi=200,
    bbox_inches="tight"
)

print("已保存: outputs/004_different_layers.png")

plt.show()

# ---------------------------------
# 实验C：统计每层最后一个Token关注谁
# ---------------------------------

print("\n最后一个Token的Attention分布")

for layer_idx in range(num_layers):

    matrix = attentions[layer_idx][
        0
    ].detach().float().cpu()

    # 形状：[heads, query_len, key_len]
    # 取最后一个Query位置，得到每个Head的Key分布
    last_query = matrix[:, -1, :]

    # 对所有Head求平均
    mean_attention = last_query.mean(dim=0)

    top_values, top_indices = torch.topk(
        mean_attention,
        k=min(3, len(tokens))
    )

    print(f"\nLayer {layer_idx}:")

    for value, idx in zip(
        top_values.tolist(),
        top_indices.tolist()
    ):
        print(
            f"  Key位置={idx}, "
            f"Token={labels[idx]!r}, "
            f"平均权重={value:.4f}"
        )
