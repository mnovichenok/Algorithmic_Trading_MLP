from quay.io/lib/python

COPY requirements.txt requirements.txt
COPY fast_api.py fast_api.py
COPY MLP.py MLP.py
COPY indicator.py indicator.py

RUN pip install -r requirements.txt --no-cache-dir

CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]
