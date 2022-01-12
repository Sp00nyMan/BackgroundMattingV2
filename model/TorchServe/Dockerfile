FROM pytorch/torchserve:latest

COPY . /home/model-server/
#WORKDIR /home/model-server/
#
#ENV PYTHONPATH /home/model-server/

USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN chown model-server:model-server -R /home/model-server
USER model-server

RUN torch-model-archiver   --model-name=test   \
    --version=1.0   \
    --model-file=/home/model-server/model.py   \
    --serialized-file=/home/model-server/pytorch_resnet101.pth   \
    --handler=/home/model-server/torchserve_handler.py   \
    --export-path=/home/model-server/model-store

CMD ["torchserve", \
    "--start", \
    "--ts-config=/home/model-server/config.properties", \
    "--model-store=/home/model-server/model-store", \
    "--models", "test=test.mar"]