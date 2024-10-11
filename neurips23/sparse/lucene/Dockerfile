FROM neurips23

RUN apt update
RUN apt-get install -y python-setuptools python-pip
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
RUN pip3 install JPype1

COPY lucene-test/lucene-core-9.11.1.jar /home/app/lucene-core-9.11.1.jar
COPY lucene-test/lucene-search-engine.jar /home/app/lucene-search-engine.jar

WORKDIR /home/app