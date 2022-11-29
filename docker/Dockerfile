FROM tensorflow/tensorflow:2.11.0-gpu AS gpu
COPY ./ /dist
RUN pip install --upgrade pip && pip install --no-cache-dir /dist/*.whl && rm -R /dist
RUN mkdir -p /home/.config/ && chmod -R a+rwx /home/
ENV MPLCONFIGDIR "/home/.config"
WORKDIR /home/data

FROM tensorflow/tensorflow:2.11.0 AS cpu
COPY . /dist
RUN pip install --upgrade pip && pip install --no-cache-dir /dist/*.whl && rm -R /dist
RUN mkdir -p /home/.config/ && chmod -R a+rwx /home/
ENV MPLCONFIGDIR "/home/.config"
WORKDIR /home/data
