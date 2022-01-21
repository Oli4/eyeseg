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
build_gpu:
	DOCKER_BUILDKIT=1 docker build -t medvisbonn/octseg:0.1-gpu -f Dockerfile --target gpu ../

build_cpu:
	DOCKER_BUILDKIT=1 docker build -t medvisbonn/octseg:0.1-cpu -f Dockerfile --target cpu ../

build_all: build_gpu build_cpu


bash:
	docker run $(USER) -it $(GPUFLAG) -v $(DATA):/home/data medvisbonn/octseg:0.1-$(COMPUTETAG)

docker run -u $(id -u):$(id -g) -v YOUR_DATA_PATH:/home/data -it medvisbonn/octseg:0.1-cpu
