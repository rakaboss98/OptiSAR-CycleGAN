# getting base image
FROM ubuntu
COPY . /usr/optisar-cyclegan
EXPOSE 5000
WORKDIR /usr/optisat-cyclegan
RUN apt-get update
RUN pip install -r requirements.txt
CMD ["echo", "Created the docker image successfully"]

