from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "Qwen/Qwen3-0.6B"


print("正在加载模型...")


tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


model = AutoModelForCausalLM.from_pretrained(
    model_name
)


print("模型加载完成")


prompt = "请解释Transformer中的Attention机制"


inputs = tokenizer(
    prompt,
    return_tensors="pt"
)


outputs = model.generate(
    **inputs,
    max_new_tokens=200
)


result = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)


print(result)
