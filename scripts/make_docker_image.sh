poetry build
DOCKER_BUILDKIT=1 docker build -t medvisbonn/octseg:0.1-gpu -f ./docker/Dockerfile --target gpu .
docker save -o ./dist/octseg_gpu_docker.tar medvisbonn/octseg:0.1-cpu
gzip ./dist/octseg_gpu_docker.tar
