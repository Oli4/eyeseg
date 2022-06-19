poetry build
DOCKER_BUILDKIT=1 docker build -t medvisbonn/eyeseg:0.1-gpu -f ./docker/Dockerfile --target gpu .
#docker save -o ./dist/docker_octseg-0.2-gpu.tar medvisbonn/eyeseg:0.1-gpu
#gzip ./dist/docker_eyeseg-0.1-gpu.tar
