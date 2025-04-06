### Install requirements with
# conda create --name lts-3.9-prod python=3.9
# conda activate lts-3.9-prod
# pip3 install poetry
# poetry install


scrape-lunar-pit-atlas:
	@python3 src/manual_scripts/scrape_pit_atlas.py

worker-build:
	@docker build -f Dockerfile.worker -t worker .

worker-start:
	@mkdir -p ${UTILITY_VOLUME}
	@chmod -R 777 ${UTILITY_VOLUME}
	@docker run -it --rm \
		--env-file .env \
		--network=host \
		-v $(UTILITY_VOLUME):/app/data \
		-u $(shell id -u):$(shell id -g) \
  		worker || true

format:
	@poetry run black . --quiet

lint:
	@poetry run pylint src || true

typecheck:
	@poetry run mypy src || true
