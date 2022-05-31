# Base image from pytorch
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
# Set up for your local zone an UTC information
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Additional librairies
RUN pip install tqdm tensorboard gradio