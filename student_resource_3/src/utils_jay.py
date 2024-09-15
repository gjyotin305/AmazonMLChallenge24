import os
import pandas as pd
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from utils import download_images
from constants import entity_unit_map
from PIL import Image
from tqdm import tqdm

def predict(image_path: str, entity_name: str) -> str:
    
    pass

DATASET_FOLDER = "../dataset/"
TRAIN_IMAGE_PATH = "/data/.jyotin/AmazonMLChallenge24/student_resource 3/images_train/"

train = pd.read_csv(
    os.path.join(DATASET_FOLDER, 'train_local.csv')
)

index = 7
sample_image = train["image_path"][index]
sample_open = Image.open(sample_image)
predict_out = train["entity_name"][index]

print(sample_image)
print(train["entity_value"][index])
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

processor.tokenizer.padding_side = "left"

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
) 

model.to("cuda:0")
model = torch.compile(model=model)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"}, 
            {"type": "text", "text": "Extract all the numerical information with appropriate desgnation."},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(prompt, sample_open, return_tensors="pt").to("cuda:0")

output = model.generate(**inputs, max_new_tokens=700)

decoded_output = processor.decode(output[0], skip_special_tokens=True)
print(str(decoded_output).split("[/INST]")[-1])

conversation_riyal = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f" \n {decoded_output} Output the information relevant to {predict_out} in the text given above, output only information and integers should be upto 3 decimal.Output the unit in form of unit mentioned in {entity_unit_map} along with numbers and it should be in form 'integer'+'unit'"},
        ],
    },
]


prompt1 = processor.apply_chat_template(conversation_riyal, add_generation_prompt=True)
inputs1 = processor(prompt1, return_tensors="pt").to("cuda:0")

output1 = model.generate(**inputs1, max_new_tokens=300)


decoded_output1 = processor.decode(output1[0], skip_special_tokens=True)

print(str(decoded_output1).split("[/INST]")[-1])
