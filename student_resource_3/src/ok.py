from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def generate_mistral(prompt: str) -> str:     
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    conversation = [
        {"role": "user", "content": f"{prompt}"}
    ]

    # format and tokenize the tool use prompt 
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True
    )

    model = model.to("cuda")
    model = torch.compile(model=model)

    inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    result_ = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result_
