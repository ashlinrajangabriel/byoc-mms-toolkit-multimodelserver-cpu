# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set Docker labels to enable SageMaker capabilities
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Update and upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Install software-properties-common to manage repositories
RUN apt-get install -y software-properties-common

# Add the Deadsnakes PPA for newer Python versions
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.11 and verify the installation
RUN apt-get update && apt-get install -y python3.11 && python3.11 --version

# Set Python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --set python /usr/bin/python3.11

# Install essential packages
RUN apt-get -y install --no-install-recommends build-essential ca-certificates openjdk-8-jdk-headless curl vim && \
    rm -rf /var/lib/apt/lists/*

# Install pip using get-pip script
RUN curl -O https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py

# Install necessary Python libraries
RUN pip --no-cache-dir install pandas==2.1.4 scikit-learn==1.2.2 multi-model-server sagemaker-inference retrying

# Set up directories and copy necessary files
RUN mkdir -p /home/model-server/
COPY application/sampleusecase/inference/model_handler.py /home/model-server/model_handler.py
COPY application/sampleusecase/dockerd-entrypoint.py /usr/local/bin/
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

# Define the entrypoint and command
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]
CMD ["serve"]
