import os
import pyembroidery
import pandas as pd
from datasets import load_dataset, Dataset, Features
import datasets
from PIL import Image
from transformers import (
  LlavaForConditionalGeneration,
  AutoProcessor,
)

def load_image(image_path):
  return Image.open(image_path)

def load_embroidery(embroidery_path):
  return repr(pyembroidery.read_dst(embroidery_path))

def load_dataset(data_dir):
  dataset = []
  for image_file in os.listdir(f"{data_dir}/preview"):
    image_file = f"{data_dir}/preview/{image_file}"
    embroidery_file = image_file.replace(".png", ".dst").replace("/preview/", "/embroidery/")
    feature = datasets.Image(decode=False)
    dataset.append({
        "image": feature.encode_example(load_image(image_file)),
        "embroidery": load_embroidery(embroidery_file)
    })
  return dataset

# Configure features (type and shape)
features = Features({
  "embroidery": datasets.Value("string"),
  "image": datasets.Image(),
})

# Create dataset
dataset = Dataset.from_pandas(pd.DataFrame(load_dataset("data/datasets/dst")), features=features)
print(dataset)

# Load model
checkpoint = "Intel/llava-gemma-2b"
model = LlavaForConditionalGeneration.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)

# Prepare inputs
# Use gemma chat template
prompt = processor.tokenizer.apply_chat_template(
    [{'role': 'user', 'content': "<image>\nConvert to .dst file"}],
    tokenize=False,
    add_generation_prompt=True
)
image = dataset[0]["image"]
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_length=30)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)

# TODO: training
