# EmptyAPI

API for a machine learning model trained to detect empty pages from scanned document images. 
The user sends the API an input image (in .jpg, .png or .tiff format) of a scanned document, and the API returns a reply 
containing the predicted classification ('ok' or 'empty'), and the corresponding prediction confidence (a number
between 0 and 1).   

## Model training and testing 

The neural network model used for the image classification task was built using the PyTorch library, and the model training
was done by fine-tuning an existing [Densenet neural network model](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html).
The trained model file was transformed into the [ONNX](https://onnx.ai/) format in order to speed up inference and to make the use of the model less dependent on specific frameworks and libraries. 

Class|Training samples|Validation samples|Test samples|Test accuracy
-|-|-|-|-
Post-it|3400|425|425|98.80%
No post-it|41 377|5133|4829|99.75%

The model has been trained and tested using 55 545 scanned document images, out of which 4250 images contain post-it notes. With a test set of over 5 000 images, the model reaches over 98% detection accuracy for both classes. 

The data used in model training and testing consists of documents produced by the Finnish public administration in the period from 1970s till 2020s and digitized by the 
Finnish National Archives. The documents contain sensitive data, and therefore the dataset can not be made publicly available.

## Running the API

The API code has been built using the [FastAPI](https://fastapi.tiangolo.com/) library. It can be run either in a virtual environment,
or in a Docker container. Instructions for both options are given below. 

The API uses the pretrained machine learning model file located in the `/model` folder. By default the file name should be `post_it_model.onnx`.
If you use a model with different name, you need to update the model name in the `MODEL_PATH` variable of the `api.py` file.

### Running the API in a virtual environment

These instructions use a conda virtual environment, and as a precondition you should have Miniconda or Anaconda installed on your operating system. 
More information on the installation is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 

#### Create and activate conda environment using the following commands:

`conda create -n postit_api_env python=3.7`

`conda activate postit_api_env`

#### Install dependencies listed in the *requirements.txt* file:

`pip install -r requirements.txt`

#### Start the API running a single process (with Uvicorn server):

Using default host: 0.0.0.0, default port: 8000

`uvicorn api:app`
 
Select different host / port:

`uvicorn api:app --host 0.0.0.0 --port 8080`

#### You can also start the API with Gunicorn as the process manager (find more information [here](https://fastapi.tiangolo.com/deployment/server-workers/)):

`gunicorn api:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080`

  - workers: The number of worker processes to use, each will run a Uvicorn worker

  - worker-class: The Gunicorn-compatible worker class to use in the worker processes

  - bind: This tells Gunicorn the IP and the port to listen to, using a colon (:) to separate the IP and the port

### Running the API using Docker

As a precondition, you should have Docker Engine installed. More information on the installation can be found [here](https://docs.docker.com/engine/install/). 

#### Build Docker image using the *Dockerfile* included in the repository: 

`docker build -t postit_image .`

Here the new image is named postit_image. After successfully creating the image, you can find it in the list of images by typing `docker image ls`.

#### Create and run a container based on the image:

`sudo docker run -d --name postit_container -p 8000:8000 postit_image`

In the Dockerfile, port 8000 is exposed, meaning that the container listens to that port. In the above command, the corresponding host port can be chosen as the first element in `-p <host-port>:<container-port>`. If only the container port is specified, Docker will automatically select a free port as the host port. 
The port mapping of the container can be viewed with the command `sudo docker port postit_container`

## Testing the API

The API has two endpoints: `/postit` endpoint expects the input image to be included in the client's POST request, while  
`/postiturl` endpoint expects to receive the filepath to the image as a query parameter.

### Testing the API in a virtual environment

You can test the `/postit` endpoint of the API for example using curl:

`curl http://127.0.0.1:8000/postit -F file=@/path/img.jpg`

The second option is to send the url/path to the image file with the http request:

`curl http://127.0.0.1:8000/postiturl?url=/path/img.jpg`

The host and port should be the same ones that were defined when starting the API.
The image path `/path/img.jpg` should be replaced with a path to the image that is used as test input. 

### Testing the API using Docker

In the Docker version of the API, it is easiest to use the `/postit` endpoint of the API. This can be tested 
for example using curl:

`curl http://127.0.0.1:8000/postit -F file=@/path/img.jpg`

Sending the url/path to the image file with the http request to the API requires 
the use of [bind mount](https://docs.docker.com/storage/bind-mounts/) to mount the desired file or 
directory into the Docker container. For instance if the input images are located in a local folder 
`/home/user/data`, the container can be created and started the using the command 

`docker run -v /home/user/data:/data -d --name postit_container -p 8000:8000 postit_image`

and then the image paths can be sent to the API with the http request:

`curl http://127.0.0.1:8000/postiturl?url=/data/img.jpg`

### Output of the API

The output is in a .json form and consists of the predicted class label and the confidence for the prediction.
So for instance the output could be 

`{"prediction":"post-it","confidence":0.995205283164978}`


