FROM python:3.8.5

RUN apt-get update
RUN apt-get install -y default-jre 
RUN apt-get install -y default-jdk 

RUN mkdir /opt/application

COPY ./Application.py /opt/application/Application.py
COPY ./Localizator.py /opt/application/Localizator.py
COPY ./Patches_localizator.py /opt/application/Patches_localizator.py
COPY ./requirements.txt /opt/application/requirements.txt
COPY ./SafetyCheck.py /opt/application/SafetyCheck.py
COPY ./SafetyCheckUtil.py /opt/application/SafetyCheckUtil.py
COPY ./Scratch_localizator.py /opt/application/Scratch_localizator.py
COPY ./server.py /opt/application/server.py
COPY ./aiva.jar /opt/application/aiva.jar
COPY ./MyCanvas.class /opt/application/MyCanvas.class
COPY ./server.sh /opt/application/server.sh
RUN mkdir /opt/application/CNN_UTIL
COPY ./CNN_UTIL/weights_improvement.52-0.0150.h5 /opt/application/CNN_UTIL/weights_improvement.52-0.0150.h5
COPY ./CNN_UTIL/mi_test.csv /opt/application/CNN_UTIL/mi_test.csv
COPY ./NEU-DET /opt/application/NEU-DET

WORKDIR /opt/application
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD /opt/application/server.sh & bash
