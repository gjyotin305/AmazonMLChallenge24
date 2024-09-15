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
TRAIN_IMAGE_PATH = "/data/.jyotin/AmazonMLChallenge24/student_resource_3/images_train/"

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
            {"type": "image"},  # Ensure image is processed correctly if used
            {"type": "text", "text": "Extract all the information."},
        ],
    },
]

# Apply the chat template and generate input tensors
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(prompt, sample_open, return_tensors="pt").to("cuda:0")

# Generate the output using the model
output = model.generate(**inputs, max_new_tokens=300)

# Decode the output
decoded_output = processor.decode(output[0], skip_special_tokens=True)

# Construct the next conversation without including an image, focusing on the text
conversation_riyal = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": decoded_output},
            {"type": "text", "text": f"Output the information relevant to {predict_out}, output in only 2 words"},
        ],
    },
]

# Apply the chat template and generate input tensors for conversation_riyal
prompt1 = processor.apply_chat_template(conversation_riyal, add_generation_prompt=True)
inputs1 = processor(prompt1, return_tensors="pt").to("cuda:0")

# Generate the second output
output1 = model.generate(**inputs1, max_new_tokens=300)

# Decode the second output
decoded_output1 = processor.decode(output1[0], skip_special_tokens=True)

# Print the final output
print(decoded_output1)
