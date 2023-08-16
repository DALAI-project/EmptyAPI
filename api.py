from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms
import onnxruntime
import numpy as np
import io
from PIL import Image
import uvicorn
import logging
import sys
import requests

# For logging options see
# https://docs.python.org/3/library/logging.html
logging.basicConfig(filename='api_log.log', filemode='w', format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

# Path to pretrained model
MODEL_PATH = './model/empty_model_v4.onnx'
# Input image size
IMG_SIZE = 224

# Transformations used for input images
img_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Predicted class labels
classes = {0: 'empty', 1: 'ok' }

try:
    # Initialize API Server
    app = FastAPI()
except Exception as e:
    logging.error('Failed to start the API server: %s' % e)
    sys.exit(1)


# Function is run (only) before the application starts
@app.on_event("startup")
async def load_model():
    """
    Load the pretrained model on startup.
    """
    try:
        # Load the onnx model and the trained weights
        model = onnxruntime.InferenceSession(MODEL_PATH)
        # Add model to app state
        app.package = {"model": model}
    except Exception as e:
        logging.error('Failed to load the model file: %s' % e)
        raise HTTPException(status_code=500, detail='Failed to load the model file: %s' % e)

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def predict(image):
    """
    Perform prediction on input image.
    """
    # Get model from app state
    model = app.package["model"]
    image = img_transforms(image.convert("RGB")).unsqueeze(0)
    # Transform tensor to numpy array
    img = image.detach().cpu().numpy()
    input = {model.get_inputs()[0].name: img}
    # Run model prediction
    output = model.run(None, input)
    # Get predicted class
    pred = np.argmax(output[0], 1)
    pred_class = pred.item()
    # Get the confidence value for the prediction
    pred_confidences = softmax(output[0][0])[pred_class]
    # Confidence of the prediction as %
    pred_confidences = float(pred_confidences)
    # Return predicted class and confidence in dictionary form
    predictions = {'prediction': classes[pred_class], 'confidence': pred_confidences}

    return predictions


# Endpoint for POST requests: input image is received with the http request
@app.post("/empty")
async def postit(file: UploadFile = File(...)):
    try:
        # Loads the image sent with the POST request
        req_content = await file.read()
        image = Image.open(io.BytesIO(req_content)).convert('RGB')
        image.draft('RGB', (IMG_SIZE, IMG_SIZE))
    except Exception as e:
        logging.error('Failed to load the input image file: %s' % e) 
        raise HTTPException(status_code=400, detail='Failed to load the input image file: %s' % e)

    # Get predicted class and confidence
    try: 
        predictions = predict(image)
    except Exception as e:
        logging.error('Failed to analyze the input image file: %s' % e)
        raise HTTPException(status_code=500, detail='Failed to analyze the input image file: %s' % e)
        
    return predictions

# Endpoint for GET requests: input image path is received with the http request
@app.get("/emptypath")
async def postit_url(path: str):
    try:
        # Loads the image from the path sent with the GET request
        image = Image.open(path).convert('RGB')
        image.draft('RGB', (IMG_SIZE, IMG_SIZE))

    except Exception as e:
        logging.error('Failed to recognize file %s as an image. Error: %s' % (path, e))
        raise HTTPException(status_code=400, detail='Failed to load the input image file: %s' % e)

    # Get predicted class and confidence
    try: 
        predictions = predict(image)
    except Exception as e:
        logging.error('Failed to analyze the input image file: %s' % e)
        raise HTTPException(status_code=500, detail='Failed to analyze the input image file: %s' % e)

    return predictions

# Endpoint for GET requests: input image path is received with the http request
@app.get("/emptyurl")
async def postit_url(url: str):
    try:
        # Loads the image from the path sent with the GET request
        req_content = requests.get(url)
        image = Image.open(io.BytesIO(req_content.content)).convert('RGB')
        image.draft('RGB', (IMG_SIZE, IMG_SIZE))

    except Exception as e:
        logging.error('Failed to recognize file %s as an image. Error: %s' % (url, e))
        raise HTTPException(status_code=400, detail='Failed to load the input image file: %s' % e)

    # Get predicted class and confidence
    try: 
        predictions = predict(image)
    except Exception as e:
        logging.error('Failed to analyze the input image file: %s' % e)
        raise HTTPException(status_code=500, detail='Failed to analyze the input image file: %s' % e)

    return predictions

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
