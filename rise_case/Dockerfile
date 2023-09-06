FROM python:3.9-slim

COPY requirements.txt /usr/local/rise-api/requirements.txt

RUN pip install -r /usr/local/rise-api/requirements.txt

COPY . /usr/local/rise-api

WORKDIR /usr/local/rise-api

ADD . /usr/local/rise-api

ENTRYPOINT [ "python3" ]

EXPOSE 8000

CMD [ "api.py" ]