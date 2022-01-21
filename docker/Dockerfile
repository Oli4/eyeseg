FROM tensorflow/tensorflow:2.7.0-gpu AS gpu
COPY ./dist /dist
RUN pip install --upgrade pip && pip install --no-cache-dir /dist/*.whl && rm -R /dist
RUN mkdir -p /home/.config/ && chmod -R a+rwx /home/
ENV MPLCONFIGDIR "/home/.config"

FROM tensorflow/tensorflow:2.7.0 AS cpu
COPY ./dist /dist
RUN pip install --upgrade pip && pip install --no-cache-dir /dist/*.whl && rm -R /dist
RUN mkdir -p /home/.config/ && chmod -R a+rwx /home/
ENV MPLCONFIGDIR "/home/.config"
