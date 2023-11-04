FROM python:3.10.13-slim-bookworm

RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["src/predict.py", "src/preprocessing.py", "./"]
COPY ["models/1.0-xgboost.bin", "models/"]
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]