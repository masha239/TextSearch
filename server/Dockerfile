FROM gcc:latest
MAINTAINER Mariia Vedernikova "masha239@gmail.com"

RUN apt-get update && apt-get install -y cmake

# Flask
RUN apt-get update -y
RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3-pip

COPY src app
WORKDIR /app
COPY model/model.onnx ./model.onnx
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "app.py", "--model", "model.onnx"]
