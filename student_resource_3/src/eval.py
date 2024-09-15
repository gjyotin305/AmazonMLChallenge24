import os
import pandas as pd
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ok import generate_mistral
from utils import download_images
from constants import entity_unit_map
from typing import Dict
from tqdm import tqdm
import gc
import json
from PIL import Image
from tqdm import tqdm

def predict(
    processor,
    model,
    image_var : Image, 
    predict_out: str, 
    entity_unit_map: Dict[str, str]
) -> str:  
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": f"Extract all information with appropriate designation"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        conversation, 
        add_generation_prompt=True
    )
    inputs = processor(prompt, image_var, return_tensors="pt").to("cuda:0")

    output = model.generate(**inputs, max_new_tokens=700)

    decoded_output = processor.decode(output[0], skip_special_tokens=True)
    print(str(decoded_output).split("[/INST]")[-1])

    decoded_out = str(decoded_output).split("[/INST]")[-1]
    prompt_model = f" \n {decoded_out} \nOutput the information relevant to {predict_out} in the text given above, output only information and integers should be upto 3 decimal places.Output only one answer in form of the proper unit mentioned in {entity_unit_map[predict_out]}. Output in json format with unit as key and value."

    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    
    conversation = [
        {"role": "user", "content": f"{prompt_model}"}
    ]

    # format and tokenize the tool use prompt 
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_mistral = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    model_mistral = model_mistral.to("cuda")

    model_mistral = torch.compile(model=model_mistral)

    inputs.to("cuda")
    outputs = model_mistral.generate(**inputs, max_new_tokens=100)
    
    result_ = tokenizer.decode(outputs[0])

    result_dict = str(result_).split("<|assistant|>")[-1]

    result_dict_ = str(result_dict).split("<|end|>")[0]

    if "json" in result_dict_ and "```" in result_dict_:
        result_intermediate = result_dict_.split("json")[1]
        result_final = result_intermediate.split("```")[0]
        
    else:
        result_final = result_dict_    

    print(result_final)

    return result_final

DATASET_FOLDER = "../dataset/"
TRAIN_IMAGE_PATH = "/data/.jyotin/AmazonMLChallenge24/student_resource 3/images_train/"

train = pd.read_csv(
    os.path.join(DATASET_FOLDER, 'test_local.csv')
)


processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

processor.tokenizer.padding_side = "left"

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
) 

model.to("cuda:0")
model = torch.compile(model=model)

predicted = []
indexes = []

for index in tqdm(range(len(train.index))):
    sample_image = train["image_path"][index]
    sample_open = Image.open(sample_image)
    predict_out = train["entity_name"][index]

    indexes.append(index)

    print(sample_image)

    final_json = predict(
        processor=processor, 
        model=model, 
        image_var=sample_open, 
        predict_out=predict_out, 
        entity_unit_map=entity_unit_map
    )

    try:
        data = json.loads(final_json)
        print("JSON parsed successfully:", data)
        for key, val in data.items():
            predicted.append(f"{val} {key}")
            break
    except json.JSONDecodeError as e:
        data = ""
        predicted.append(data)
        print("Error parsing JSON:", e)
    
    gc.collect()
    torch.cuda.empty_cache()

    if index == 10:
        break

new_df = pd.DataFrame()
new_df["index"] = indexes
new_df["prediction"] = predicted

new_df.to_csv("../dataset/test_out.csv", index=False)