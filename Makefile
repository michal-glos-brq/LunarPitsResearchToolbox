### Install requirements with
# conda create --name lts-3.9-prod python=3.9
# conda activate lts-3.9-prod
# pip3 install poetry
# poetry install


scrape-lunar-pit-atlas:
	@python3 src/scripts/scrape_pit_atlas.py.py

worker-build:
	@docker build -f Dockerfile.worker -t worker .

worker-start:
	@mkdir -p ${UTILITY_VOLUME}
	@chmod -R 777 ${UTILITY_VOLUME}
	@docker run -it --rm \
		-e WORKER_ID=${WORKER_ID} \
		-v $(UTILITY_VOLUME):/app/data \
  		worker || true
