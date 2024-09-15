import os
import pandas as pd
from utils import download_images
from tqdm import tqdm

DATASET_FOLDER = "../dataset/"
TRAIN_IMAGE_PATH = "/data/.jyotin/AmazonMLChallenge24/student_resource_3/image_test/"

train = pd.read_csv(
    os.path.join(DATASET_FOLDER, 'train.csv')
)
test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

train_image_path = []

for x in test["image_link"]:
    url_pp = x.split("/")[-1]
    train_image_path.append(f"{TRAIN_IMAGE_PATH+url_pp}")

test["image_path"] = train_image_path

test.to_csv("../dataset/test_local.csv")