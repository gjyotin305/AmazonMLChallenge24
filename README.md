# Amazon ML Challenge 2024

Welcome to the Amazon ML Challenge 2024 repository! This project showcases solutions and implementations for the machine learning challenge organized by Amazon, focusing on various aspects of image analysis and processing.

## Project Overview

This repository contains code and resources for solving the Amazon ML Challenge 2024. The challenge involves predicting specific attributes from images using advanced machine learning models and techniques.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Data](#data)
4. [Model](#model)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/gjyotin305/AmazonMLChallenge24.git
    ```

2. Navigate to the project directory:

    ```bash
    cd AmazonMLChallenge24
    ```

3. Create a virtual environment and activate it:

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use: env\Scripts\activate
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To download the images run:

```bash
python student_resource_3/src/download_data.py
```

To further add the image path to the csv

```bash
python preprocess.py
```

To run the image prediction and processing pipeline:

1. Ensure you have the dataset available at the specified paths or adjust the paths in the script accordingly.

2. Run the main script:

    ```bash
    python eval.py
    ```

   This script will process images and output predictions as defined in the code.

3. For specific tasks or stages, you can modify and execute scripts located in the `scripts` folder.

## Data

The project uses images and related data for training and testing. Ensure that you have the dataset in the following directory structure:

- `../dataset/`: Contains CSV files and other metadata.
- `/data/.jyotin/AmazonMLChallenge24/student_resource 3/images_train/`: Contains training images.

## Model

The project utilizes the `LlavaNextForConditionalGeneration` model from Hugging Face’s Transformers library. The model is fine-tuned for the specific task of extracting and analyzing numerical information from images.

- **Model Name**: `llava-hf/llava-v1.6-mistral-7b-hf`
- **Processor**: `LlavaNextProcessor`

After we get the image description from LlavaNext, we then used a 

- **Model Name**: `Phi3-Mini-4K Instruct`, 

to further process the relevant text in a relevant json, which is then checked to ensure proper units and values are only allowed.

We load the model in float16 to ensure better execution speeds.

Our approach utilises only `10 sec` per image.

## Results

Results of the predictions and analyses are printed to the console. Modify the script to save results to files or visualize them as needed.


## Team Members

Meet our team members:

1. Rhythm Baghel.
2. Jyotin Goel.
3. Harshiv Shah.
4. Mehta Jay Kamalkumar.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
