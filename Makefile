### Install requirements with
# conda create --name lts-3.9-prod python=3.9
# conda activate lts-3.9-prod
# pip3 install poetry
# poetry install


scrape-lunar-pit-atlas:
	@python3 src/data_fetchers/lunar_pit_atlas_fetcher.py

pythonpath:
	@export PYTHONPATH=$PYTHONPATH:$(PWD)

# Join the shared network
zerotier:
	@sudo zerotier-cli join ${NETWORK_ID}

# Install zerotier software
zerotier-install:
	@curl -s https://install.zerotier.com | sudo bash


worker-build:
	@docker build -f Dockerfile.worker -t worker .

worker-start:
	@mkdir -p ${UTILITY_VOLUME}
	@chmod -R 777 ${UTILITY_VOLUME}
	@docker run -it --rm \
  		--cap-add=NET_ADMIN \
  		--cap-add=SYS_ADMIN \
  		--device=/dev/net/tun \
  		-e ZEROTIER_JOIN_ID=${NETWORK_ID} \
		-e GH_TOKEN=${GH_TOKEN} \
		-e WORKER_ID=${WORKER_ID} \
		-v $(UTILITY_VOLUME):/app/data \
  		worker || true

### This will be eventually done through a python script in a loop

# run-celery-worker: setup-pythonpath
# 	celery -A src.celery.app.app  worker -l info -P gevent --autoscale=$(LOW),$(HIGH) --hostname=worker-localhost

# run-celery-worker-debug: setup-pythonpath
# 	celery -A src.celery.app.app  worker -l debug -P gevent --autoscale=$(LOW),$(HIGH) --hostname=worker-localhost

# run-flower: setup-pythonpath
# 	@celery -A src.celery.app.app flower --port=5555
