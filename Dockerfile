FROM       tensorflow/tensorflow:latest

EXPOSE     5000

WORKDIR    /app
COPY       requirements.txt /app/
RUN        pip install -r requirements.txt



COPY       *.py /app
COPY       *.h5 /app
COPY       *.pkl /app
COPY       *.json /app
RUN        chmod a+x *.py

CMD        ["./main.py"]
