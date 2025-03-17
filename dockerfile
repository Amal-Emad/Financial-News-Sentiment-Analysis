# Use an official Python runtime as base image
FROM python:3.10


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 5000


CMD ["python", "src/app.py"]

