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

sample_image = train["image_path"][1]
sample_open = Image.open(sample_image)
predict_out = train["entity_name"][1]

print(sample_image)

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
            {"type": "text", "text": "Extract all the information."},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(prompt, sample_open, return_tensors="pt").to("cuda:0")

output = model.generate(**inputs, max_new_tokens=300)

decoded_output = processor.decode(output[0], skip_special_tokens=True)

conversation_riyal = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": decoded_output},
            {"type": "text", "text": f"Output the information relevant to {predict_out}"},
        ],
    },
]


print(f"Second conversation (conversation_riyal): {conversation_riyal}")