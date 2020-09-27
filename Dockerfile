FROM python:3.6

ADD . /opt/project
WORKDIR /opt/project

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt