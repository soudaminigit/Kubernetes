FROM python:3.10-slim

WORKDIR /app
RUN pip install scikit-learn

COPY train.py train.py

CMD ["python", "train.py"]
