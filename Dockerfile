FROM pytorch/pytorch:latest

WORKDIR /workspace
ADD . /workspace

RUN pip install -r requirements.txt

RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace
EXPOSE 8080
CMD [ "python3" , "/workspace/app.py" ]
