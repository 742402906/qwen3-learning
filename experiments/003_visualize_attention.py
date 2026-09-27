import torch
import matplotlib.pyplot as plt
import numpy as np

# macOS中文字体，避免图中中文显示为方框
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Arial Unicode MS",
    "Hiragino Sans GB",
    "Heiti TC",
]
plt.rcParams["axes.unicode_minus"] = False

from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================
# 1. 实验配置
# ==========================================

MODEL_NAME = "Qwen/Qwen3-0.6B"

TEXT = "什么是AI测试"

# Transformer层，从0开始
LAYER_INDEX = 0

# Attention Head，从0开始
HEAD_INDEX = 0

# 输出图片路径
OUTPUT_PATH = "outputs/attention_layer0_head0.png"


# ==========================================
# 2. 选择设备
# ==========================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("使用设备:", device)


# ==========================================
# 3. 加载Tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ==========================================
# 4. 加载模型
# ==========================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    attn_implementation="eager"
)

model.to(device)
model.eval()

print("模型加载完成")


# ==========================================
# 5. Tokenize
# ==========================================

inputs = tokenizer(
    TEXT,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

input_ids = inputs["input_ids"][0]

tokens = tokenizer.convert_ids_to_tokens(
    input_ids.tolist()
)

print("\n输入文本:", TEXT)

print("\nToken数量:", len(tokens))

print("\nToken列表:")

for i, (token_id, token) in enumerate(
    zip(input_ids.tolist(), tokens)
):
    print(
        f"位置={i}, "
        f"ID={token_id}, "
        f"Token={repr(token)}"
    )


# ==========================================
# 6. 前向传播，获取Attention
# ==========================================

with torch.no_grad():

    outputs = model(
        **inputs,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True
    )


attentions = outputs.attentions

print("\nAttention层数:", len(attentions))


# ==========================================
# 7. 提取指定层和Head
# ==========================================

attention = attentions[LAYER_INDEX]

print(
    "当前层Attention形状:",
    tuple(attention.shape)
)

# Attention张量形状:
# [batch, num_heads, query_len, key_len]

attention_matrix = attention[
    0,
    HEAD_INDEX,
    :,
    :
]

# 转移到CPU，供NumPy和Matplotlib使用
attention_matrix = (
    attention_matrix
    .detach()
    .float()
    .cpu()
    .numpy()
)

print(
    "当前Head矩阵形状:",
    attention_matrix.shape
)


# ==========================================
# 8. 绘制热力图
# ==========================================

# 用Token ID对应的可读文本作为坐标标签
labels = [
    f"{i}: {tokenizer.convert_tokens_to_string([token])}"
    for i, token in enumerate(tokens)
]

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    attention_matrix,
    cmap="viridis",
    aspect="auto",
    interpolation="nearest"
)

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))

ax.set_xticklabels(
    labels,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(labels)

ax.set_xlabel("Key Token (被关注的Token)")
ax.set_ylabel("Query Token (发起关注的Token)")

ax.set_title(
    f"Qwen3-0.6B Attention\n"
    f"Layer {LAYER_INDEX}, Head {HEAD_INDEX}"
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Attention Weight")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

print("\n热力图已保存:", OUTPUT_PATH)

plt.show()


# ==========================================
# 9. 保存原始矩阵
# ==========================================

np.save(
    "outputs/attention_matrix.npy",
    attention_matrix
)

print("原始矩阵已保存")
