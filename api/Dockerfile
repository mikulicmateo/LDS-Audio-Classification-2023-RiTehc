FROM tiangolo/uvicorn-gunicorn:python3.10

WORKDIR /api

COPY requirements.txt .

RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]