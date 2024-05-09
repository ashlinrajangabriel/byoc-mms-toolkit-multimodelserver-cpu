# Model Handling Server

This server provides an API for managing machine learning multi models (mms) and performing predictions. It supports operations such as loading models, listing available models, and deleting models, as well as making predictions using loaded models.

## Setup

### Building the Docker Image

To build the Docker image for the server, run the following command:

```bash
docker build -t my-model-handler .
```

```bash
docker run -p 8080:8080 -v /path/to/models:/opt/ml/models -it my-model-handler
```

### API Endpoints
#### Load a Model
Load a model by specifying its name and the sub-directory under /opt/ml/models where it is located

```
curl --location 'http://localhost:8080/models' \
--header 'Content-Type: application/json' \
--data '{
    "model_name": "modelx",
    "url": "/opt/ml/models/india"
}'
```

#### List Models
##### List all loaded models:

```bash
curl --location 'http://localhost:8080/models' \
--header 'Accept: application/json'

```
#### Delete a Model

```bash
curl --location --request DELETE 'http://localhost:8080/models/modelx'

```

#### Make a Prediction
```bash
curl --location 'http://localhost:8080/predictions/modelx' \
--header 'Content-Type: application/json' \
--data '{
    "action": {"0": 1, "1": 1, "2": 2, "3": 2},
    "david": {"0": 1, "1": 2, "2": 2, "3": 3}
}'

```

### Docker-EntryPoint Script
The docker-entrypoint.py script is used to manage the lifecycle of the model server. It configures and starts the model server with the desired number of worker processes. If the server is started with the serve command, it initiates the model server; otherwise, it executes the specified command.

Key Features:
Retry Logic: Automatic retry logic for starting the model server in case of errors.
Worker Configuration: Configurable number of workers per model through an environment variable.
Notes
Ensure the path mappings in Docker commands match the actual locations on your host machine to avoid issues with model loading.
The server listens on port 8080, which should be open on your host machine.
Adjust the number of workers based on your load requirements and server capacity.


### Explanation of Key Components:

1. **Docker Commands**: Illustrates how to build and run the Docker image, including volume mapping for model storage.
2. **CURL Commands**: Demonstrates how to interact with the server's API to manage models and perform predictions.
3. **docker-entrypoint.py**: Describes the purpose and functionality of the entry point script used in the Docker setup.

This README aims to provide comprehensive guidance on using your model handling server, ensuring it is accessible to users with varying levels of technical expertise. Adjust the paths and commands as necessary to match your specific deployment environment.
