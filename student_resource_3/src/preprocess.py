import os
import pandas as pd
from utils import download_images
from tqdm import tqdm

DATASET_FOLDER = "../dataset/"
TRAIN_IMAGE_PATH = "/data/.jyotin/AmazonMLChallenge24/student_resource 3/images_train/"


train = pd.read_csv(
    os.path.join(DATASET_FOLDER, 'train.csv')
)
test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

train_image_path = []

for x in train["image_link"]:
    url_pp = x.split("/")[-1]
    train_image_path.append(f"{TRAIN_IMAGE_PATH+url_pp}")

train["image_path"] = train_image_path

train.to_csv("../dataset/train_local.csv")