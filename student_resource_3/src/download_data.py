import os
import pandas as pd
from utils import download_images

DATASET_FOLDER = "../dataset/"

train = pd.read_csv(
    os.path.join(DATASET_FOLDER, 'train.csv')
)

test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

sample_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))

sample_test_out = pd.read_csv(
    os.path.join(DATASET_FOLDER, 'sample_test_out.csv')
)

download_images(train['image_link'], "../images_train")
download_images(test['image_link'], "../image_test")
download_images(sample_test['image_link'], "../image_test_sample")