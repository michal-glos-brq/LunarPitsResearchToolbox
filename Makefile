### Install requirements with
# conda create --name lts-3.9-prod python=3.9
# conda activate lts-3.9-prod
# pip3 install poetry
# poetry install


export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

scrape-lunar-pit-atlas:
	@python3 src/manual_scripts/scrape_pit_atlas.py

worker-build:
	@docker build -f Dockerfile.worker -t worker .

worker-build-no-cache:
	@docker build -f Dockerfile.worker -t worker --no-cache .

worker-start:
	@mkdir -p ${UTILITY_VOLUME}
	@mkdir -p ${UTILITY_VOLUME}/logs
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

docker-clean:
	@docker image prune -f



# One af task assignement for simulation, 1st retry
# ./assign_tasks.py --task remote_sensing --config-name lunar_pit_atlas_mapping_LRO_simulation --name full_run --retry-count 1
# Dry run of dummy example run
# ./src/manual_scripts/assign_tasks.py --config-name test_lro_short_simulation --name demo_run  --task remote_sensing

# How to make intervals from the simulation runs
# ./src/manual_scripts/aggregate_simulation_intervals.py \
  --config-name lunar_pit_atlas_mapping_LRO_simulation \
  --sim-name lunar_pit_run \
  --interval-name test_lunar_pit_run \
  --threshold 5

