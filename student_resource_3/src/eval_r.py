import os
import pandas as pd
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from utils import download_images
from constants import entity_unit_map
from PIL import Image
from tqdm import tqdm
import re  # Regex library to process output

def predict(image_path: str, entity_name: str) -> str:
    pass

DATASET_FOLDER = "../dataset/"
TRAIN_IMAGE_PATH = "/data/.jyotin/AmazonMLChallenge24/student_resource_3/images_train/"

train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train_local.csv'))

index = 18
sample_image = train["image_path"][index]
sample_open = Image.open(sample_image)
predict_out = train["entity_name"][index]

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
processor.tokenizer.padding_side = "left"

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
)

model.to("cuda:0")
model = torch.compile(model=model)

# Initial conversation to extract all numerical information
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"}, 
            {"type": "text", "text": "Extract all the numerical information."},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(prompt, sample_open, return_tensors="pt").to("cuda:0")

output = model.generate(**inputs, max_new_tokens=700)

decoded_output = processor.decode(output[0], skip_special_tokens=True)

# Prepare the second conversation to focus on extracting relevant information for 'entity_name'
conversation_riyal = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"\n {decoded_output} Output the information relevant to {predict_out} in the text given above, output only information and integers should be up to 3 decimal places. Output the unit in the form of the unit mentioned in {entity_unit_map} along with numbers, and it should be in the format 'integer'+'unit (full name eg-g should be gram)'."},
        ],
    },
]

prompt1 = processor.apply_chat_template(conversation_riyal, add_generation_prompt=True)
inputs1 = processor(prompt1, return_tensors="pt").to("cuda:0")

output1 = model.generate(**inputs1, max_new_tokens=300)

decoded_output1 = processor.decode(output1[0], skip_special_tokens=True)

# Post-processing to extract only 'number + unit' like '200g'
def extract_number_unit(text):
    # Regex to capture number + unit (e.g., 200g, 1.5kg, etc.)
    match = re.search(r"(\d+\.?\d*)\s?([a-zA-Z]+)", text)
    if match:
        return f"{match.group(1)}{match.group(2)}"  # Combine number and unit without space
    return "No relevant information found"

# Process the model's output to get just the required format
final_output = extract_number_unit(decoded_output1)
print(final_output)
