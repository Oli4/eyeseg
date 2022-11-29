help:
	@cat Makefile

ROOTLESS?=False
ifeq ($(ROOTLESS), True)
	USER=
else
	USER="-u $(shell id -u):$(shell id -g)"
endif
# For rootless docker you need to specify `all` here. You can select the GPU you want to use in your code which is executed by the container. You can check which GPU are used using 'nvidia-smi'
GPU?=all
ifneq (($shell lspci | grep -i '.* vga .* nvidia .*'), )
	GPUFLAG="--gpus=${GPU}"
	COMPUTETAG=gpu
else
	COMPUTETAG=cpu
endif


# Copy all code and models to build a production image
build_eyeseg:
	poetry build

build_gpu: build_eyeseg
	DOCKER_BUILDKIT=1 docker build -t olcifer/eyeseg:0.1-gpu -f Dockerfile --target gpu ../dist
	mkdir -p ./dist
	docker save -o ./dist/docker_eyeseg-0.1-gpu.tar olcifer/eyeseg:0.1-gpu
	gzip ./dist/docker_eyeseg-0.1-gpu.tar

build_cpu:
	DOCKER_BUILDKIT=1 docker build -t olcifer/eyeseg:0.1-cpu -f Dockerfile --target cpu ../dist
	mkdir -p ./dist
	docker save -o ./dist/docker_eyeseg-0.1-cpu.tar olcifer/eyeseg:0.1-cpu
	gzip ./dist/docker_eyeseg-0.1-cpu.tar

build_all: build_gpu build_cpu
