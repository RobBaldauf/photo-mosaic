# photo-mosaic

Web Service for community-based creation of photo mosaic images.

## Run via Docker (recommended)

Download and install docker https://docs.docker.com/engine/install

Two docker files are available for the API:

- Development ([Dockerfile](Dockerfile))
  - API documentation accessible via
    [http://127.0.0.1:8111/documentation](http://127.0.0.1:8111/documentation)
  - No authentication
  - No content filtering
- Production ([Dockerfile.prod](Dockerfile.prod))
  - API Documentation disabled
  - Simple key based authentication for admin endpoints
  - NSFW content filtering (based on https://github.com/GantMan/nsfw_model)

### Development Docker

1. Build docker image

```shell
docker build -t photo-mosaic .
```

2. Run docker container. Replace <your_db_folder> with the directory that should contain
   the SQLite DB file (e.g. ~ /mosaic_db).

```shell
docker run -t -i -p 8111:8111 -v <your_db_folder>:/db photo-mosaic:latest
```

### Production Docker

1. Build docker image. Replace <your_secret> with a secret string you would like to for
   creating/validating the API_KEY.

```shell
docker build -t photo-mosaic --build-arg JWT_SECRET=<your_secret> -f Dockerfile.prod .
```

The output of the docker build should contain a line similar to this one:

```shell
API_KEY=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhcHAiOiJwaG90by1tb3NhaWMiLCJpZCI6InBob3RvLW1vc2FpYy1hZG1pbiIsImV4cCI6MTY4OTI3OTU2NH0.A80yxWZ0rjvIi98qYZ0x1pyf1l2jH4YQExW75t2rrlU
```

This is the API KEY you need to access the admin endpoints of the API.

2. Run docker container. Replace <your_db_folder> with the directory that should contain
   the SQLite DB file (e.g. ~ /mosaic_db).

```shell
docker run -t -i -p 8111:8111 -v <your_db_folder>:/db photo-mosaic:latest
```

## Run locally

If don't want to use docker you can also install and run the service locally:

### Install

Download and install python 3.7 or higher (https://www.python.org/downloads/) as well as
pip

```shell
sudo apt install python3.7
sudo apt install python3-pip
```

Install python dependencies:

```shell
pip install -r requirements.txt
```

Give permissions to executable

```shell
chmod 755 run_app.sh
```

### Update config

Edit the config file [config/.env](config/.env) or set your desired configuration via
environment variables.

- Important settings:

```shell
# api config
ENABLE_DOCUMENTATION=true # enables/disables swagger documentation
CORS_ORIGINS=["*"] # List of allowed CORS origins (by default all origins are allowed)

# auth config
ENABLE_AUTH=false # enables/disables admin endpoint authentication
JWT_SECRET= # the secret used to create/varify the API KEY

# NSFW config
ENABLE_NSFW_CONTENT_FILTER=false # enables/disables nsfw content filtering
NSFW_MODEL_PATH=nsfw_model/mobilenet_v2_140_224 # the path to the NSFW model

# db config
SQLITE_PATH=/db # path in which the sqlite db file shall be stored
```

- To create a JWT_SECRET use this script:
  [scripts/generate_api_key.py](scripts/generate_api_key.py).
- The NSFW model can be downloaded here:
  [nsfw mobilenet model](https://github.com/GantMan/nsfw_model/releases/download/1.1.0/nsfw_mobilenet_v2_140_224.zip)

### Run service

Start photo-mosaic server:

```shell
./run_prod.sh
```

### Run tests

Run fastapi endpoint tests:

```shell
python3 -m pytest
```

## Documentation

API Documentation via swagger:
[http://127.0.0.1:8111/documentation](http://127.0.0.1:8111/documentation)

## License

[GNU General Public License v3.0](LICENSE)
