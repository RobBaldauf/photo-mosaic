FROM python:3.7
WORKDIR /photo_mosaic
EXPOSE 8111

# install python opencv (workaround for dependency bug)
RUN apt-get update && apt-get install -y python3-opencv

# Install python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy config
COPY config/ ./config

# set config for dev mode
RUN sed -i '\
    s/ENABLE_DOCUMENTATION=.*/ENABLE_DOCUMENTATION=true/g; \
    s/ENABLE_AUTH=.*/ENABLE_AUTH=false/g; \
    s/ENABLE_NSFW_CONTENT_FILTER=.*/ENABLE_NSFW_CONTENT_FILTER=false/g;' config/.env

# copy code
COPY run_app.sh .
COPY VERSION .
COPY photo_mosaic/ ./photo_mosaic

# run app
ENTRYPOINT ["./run_app.sh"]
