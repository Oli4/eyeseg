FROM tensorflow/tensorflow:2.11.0-gpu AS gpu
COPY ./ /dist
RUN apt-get update && apt-get install libgl1 -y
RUN pip install --upgrade pip && pip install --no-cache-dir /dist/*.whl && rm -R /dist
RUN mkdir -p /home/.config/ && chmod -R a+rwx /home/ && mkdir -p /.eyepy && chmod -R a+rwx /.eyepy
ENV MPLCONFIGDIR "/home/.config"
WORKDIR /home/data

FROM tensorflow/tensorflow:2.11.0 AS cpu
COPY . /dist
RUN apt-get update && apt-get install libgl1 -y
RUN pip install --upgrade pip && pip install --no-cache-dir /dist/*.whl && rm -R /dist
RUN mkdir -p /home/.config/ && chmod -R a+rwx /home/ && mkdir -p /.eyepy && chmod -R a+rwx /.eyepy
ENV MPLCONFIGDIR "/home/.config"
WORKDIR /home/data
