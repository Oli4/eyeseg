poetry build
DOCKER_BUILDKIT=1 docker build -t medvisbonn/octseg:0.1-gpu -f ./docker/Dockerfile --target gpu .
docker save -o ./dist/docker_octseg-0.1-gpu.tar medvisbonn/octseg:0.1-gpu
gzip ./dist/docker_octseg-0.1-gpu.tar
