# qwen3-learning

在 Apple Silicon（M 系列 / MPS）上学习 Qwen3-0.6B 的推理与微调：从注意力可视化、隐藏态分析，到手工 PyTorch 训练循环的 LoRA 微调与前后对比。

## 环境

- macOS + Apple Silicon（MPS 加速），Python 环境基于 conda（`qwen3`）
- torch / transformers / peft / matplotlib，模型权重从 HuggingFace 自动下载并缓存（`Qwen/Qwen3-0.6B`）

```bash
conda activate qwen3
```

## 目录结构

```
data/       训练数据（JSONL，messages 格式）
experiments/  实验脚本
outputs/    实验产物（对话记录、LoRA adapter、Loss 曲线）
scripts/    通用小脚本
```

## 实验列表

| 脚本 | 内容 |
|---|---|
| `experiments/003_visualize_attention.py` | 注意力权重可视化 |
| `experiments/004_compare_attention.py` | 不同层/头的注意力对比 |
| `experiments/005_hidden_state_analysis.py` | 隐藏状态分析 |
| `experiments/005b_hidden_state_diagnosis.py` | 隐藏状态诊断 |
| `experiments/006_baseline_chat.py` | 原始模型交互式对话，回答逐条记录到 JSONL |
| `experiments/006_train_lora.py` | LoRA 微调（无 Trainer/TRL，纯 PyTorch 循环 + PEFT） |
| `experiments/006_inference_compare.py` | 重新加载 Adapter，用相同问题对比训练前后输出 |

## Experiment 006：LoRA 微调闭环

以"智能摄像头 AI 测试助手"为目标场景，完整走通 baseline → 微调 → 对比：

1. **Baseline**：`006_baseline_chat.py` 用贪心解码（`do_sample=False`）+ 关闭思考模式（`enable_thinking=False`）记录原始模型对 5 个测试业务的回答。
2. **训练数据**：`data/experiment_006_train.jsonl`，10 条 messages 格式示例；只对 Assistant 答案计算 Loss（`labels[:prompt_length] = -100`）。
3. **训练**：`006_train_lora.py`，LoRA r=8 / alpha=16，目标覆盖注意力与 MLP 全部 7 个投影层，可训练参数约 505 万（占 0.84%）；3 epochs 后平均 Loss 2.48 → 1.02。
4. **对比结论**：训练后模型对测试用例类问题稳定输出训练格式的紧凑 JSON，且不是死记硬背（措辞与训练样本不同）；但对分布外问题（原因分析、指标解释）几乎没有迁移，且出现把所有问题都套 JSON 格式的过度泛化——10 条样本只够验证流程。

## 注意

- `outputs/` 中的 `*.safetensors` 与 `tokenizer.json`（可复现的大文件）不进 git，其余小产物均保留。
- 脚本在 transformers 5.x 下运行；`apply_chat_template(tokenize=True)` 返回 `BatchEncoding`，需取 `["input_ids"]`（见 `006_train_lora.py` 的 `encode_example`）。
