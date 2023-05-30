FROM python:3.8

WORKDIR /app-docker

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY /src/simple-linear-regression-ksarafrazi/ /app-docker/

EXPOSE 5000
CMD [ "flask", "--app", "main.py", "run", "--host","0.0.0.0","--port","5000"]
