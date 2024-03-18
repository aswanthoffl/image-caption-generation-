from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering 
import torch

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Load image from file path
image_path = r"C:/Users/NETCOM/Downloads/Image-Caption-Generation-API-main/Image-Caption-Generation-API-main/Cats.jpg"
image = Image.open(image_path)

# Training
text = "How many cats are in the picture?"
label = "2"
inputs = processor(images=image, text=text, return_tensors="pt")
labels = processor(text=label, return_tensors="pt").input_ids

inputs["labels"] = labels
outputs = model(**inputs)
loss = outputs.loss
loss.backward()

# Inference
text = "How many cats are in the picture?"
inputs = processor(images=image, text=text, return_tensors="pt")
outputs = model.generate(**inputs)
print(processor.decode(outputs[0], skip_special_tokens=True))
