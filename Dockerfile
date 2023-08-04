FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install Flask
RUN pip install ultralytics
CMD ["python", "app.py"]