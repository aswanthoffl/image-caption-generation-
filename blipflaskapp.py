from flask import Flask, request, render_template
import io
import json
from PIL import Image

# Import model-related libraries here
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Load model, tokenizer, etc. (replace with actual loading code)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for image upload and caption generation
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    contents = file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Process image and generate caption
    images = [image.convert(mode="RGB")]
    inputs = processor(images, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    inputs = inputs.to(device)
    output_ids = model.generate(**inputs, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    
    return render_template('result.html', result=preds[0])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
