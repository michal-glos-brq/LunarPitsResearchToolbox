# LunarPitsResearchToolbox

A comprehensive toolkit for lunar pit mapping and simulation, developed as part of the diploma thesis in Space Applications master program on FECT-BUT. It integrates SPICE-based instrument modeling, data analysis pipelines, and simulation frameworks to process and visualize lunar remote sensing datasets.

## Table of Contents

- [Introduction](#introduction)
- [Setup and Deployment](#setup-and-deployment)
  - [Master Node (Bare‑Metal or VM)](#master-node-baremetal-or-vm)
  - [Worker Node (Docker‑only)](#worker-node-dockeronly)
  - [Programmatic Configuration Overrides](#programmatic-configuration-overrides)
- [Core Workflows](#core-workflows)
  - [Simulation (Remote-Sensing Pre-Scan)](#1-simulation-remote-sensing-pre-scan)
  - [Extraction (Pixel-Level Re-Projection)](#2-extraction-pixel-level-re-projection)
  - [Monitoring](#3-monitoring)
- [SPICE Engine](#spice-engine)
  - [Kernel Managers](#kernel-managers)
  - [Kernel Types](#kernel-types)
  - [Dynamic Kernel Managers](#dynamic-kernel-managers)
  - [Instruments](#instruments)
- [Data Connectors](#data-connectors)
- [Abstractions](#abstractions)
- [Analysis](#analysis)


## Introduction

`LunarPitsResearchToolbox` is the software implementation accompanying the master’s thesis _Advanced System for Identification of Subsurface Cavities in Lunar Pits_. It condenses the full methodological stack—geometry reconstruction, observation‑window simulation, raw‑data ingestion, reprojection, and analytics—into a single, version‑controlled Python package that can run unchanged on laptops, clusters, or containerised cloud workers/server runners.

Its highly **modular**, **mission‑agnostic** architecture requires only minor adjustments—such as adding new kernel loaders, instruments or dataset connectors—to support additional instruments, planetary bodies, or data modalities.

In the configuration for Lunar Pits research, the pipeline ingested and processed three mission‑grade datasets—LOLA, Diviner, and Mini-RF—to search for confirmed lunar pits with precision and and intent of looking for potential proofs of underlying cavities.

Under the hood the toolbox:

- orchestrates **dynamic SPICE kernels** (CK, SPK, FK, DSK…) with automatic download, shared‑lock caching, and millisecond‑scale load/unload to keep memory use bounded
    
- runs a two‑phase **simulation → extraction** loop that trimmed >99 % of irrelevant observation time in our global 2009‑2025 LRO survey, allowing us to work with experiment data in reasonably sized chunks
    
- provides **dataset connectors** for Diviner, Mini‑RF, LOLA, and GRAIL, each exposing a uniform `.query()` interface that yields NumPy/Pandas objects ready for analysis
    
- distributes workloads via **Celery** and persists results in MongoDB, supporting seamless horizontal scaling from a single debug worker to a 64‑CPU cluster
    

### Verification at Scale

The pipeline has been exercised on

- 15 years of LRO telemetry,
    
- 400+ simulation subtasks covering every confirmed pit in the 2024 Lunar Pit Atlas, and
    
- 1600+ extraction subtasks,
    

yielding geometry‑consistent thermal, radar, and altimetric dataset for 278 pits.

---

## Setup and Deployment

Configure the master orchestration node, launch containerized workers, and apply runtime overrides.

---

### Master Node (Bare‑Metal or VM)

Ensure Docker and Conda are installed on the master host.

#### 1. Launch core services

```bash
docker-compose up -d    # starts Redis, MongoDB, Flower, Netdata containers
```

#### 2. Python environment & project path
Download [Conda](https://www.anaconda.com/download)

```bash
conda create -n lunar-pits python=3.9 -y
conda activate lunar-pits
pip install poetry
poetry install

# Add project root to PYTHONPATH for manual scripts
echo "export PYTHONPATH=$(pwd):\$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
```
Poetry works by default with its own virtualenvs, suppress this behaviour with:
```bash
poetry config virtualenvs.create false --local
```
#### 3. Populate the pit catalog

```bash
make scrape-lunar-pit-atlas    # download and cache 278 pit definitions
```

#### 4. Run manual tooling

With core services running and the environment activated, invoke any script under `src/manual_scripts`:

```bash
poetry run python src/manual_scripts/assign_tasks.py \
  --task remote_sensing \
  --config-name test_lro_short_simulation \
  --name initial_sim_run
```

| Service | Port  | Description      |
| :-----: | :---: | :--------------- |
|  redis  | 6379  | Celery broker    |
| mongodb | 27017 | Results store    |
| flower  | 5555  | Celery dashboard |
| netdata | 19999 | Monitoring UI    |

_Tip:_ You can offload MongoDB to separate hosts if throughput grows.

---

### Worker Node (Docker‑only)

Each worker runs inside a container built from `Dockerfile.worker`.

#### Prerequisite
Install [Docker](https://docs.docker.com/engine/install/ubuntu/)

```bash
sudo systemctl start docker    # ensure Docker daemon is running
```

#### 1. Build the worker image

```bash
make worker-build   # builds `lunar-pits-worker:latest`
# or
make worker-build-no-cache   # without cache ...
```

#### 2. Configure per-node settings

```bash
cp .env_example .env    # edit MASTER_IP, CONCURRENCY, UTILITY_VOLUME, PDS credentials
```

#### 3. Launch and join the cluster

```bash
source .env
make worker-start
```

Workers auto‑mount the shared data volume, initialize Conda inside the container, and register the Celery `tasks` entrypoints.

---

### Programmatic Configuration Overrides

Override global settings in ad‑hoc scripts or Jupyter notebooks:

```python
import src.global_config as g
import src.db.config as db

# Redirect data storage
g.HDD_BASE_PATH = '/mnt/ssd/pits'
# Suppress progress bars
g.SUPPRESS_TQDM = True

# Point to a custom MongoDB instance
db.MONGO_URI = 'mongodb://admin:password@master:27017'
```

Use this pattern when embedding the toolbox in larger workflows or notebooks, you can configure any conf. variable without rewriting it in the project.

---

## Core Workflows

Experiment configurations live under `src/experiments/remote_sensing` and `src/experiments/extraction`. Two reference configs demonstrate how to define and register experiments for testing or as templates.

```python
# src/experiments/simulations/test_simulation_experiment.py
from astropy.time import Time
from .base_simulation_experiment import BaseSimulationConfig

class TestLROShortSimulationConfig(BaseSimulationConfig):
    experiment_name = "test_lro_short_simulation"
    instrument_names = ["DIVINER", "LOLA"]
    kernel_manager_type = "LRO"
    start_time = Time("2012-07-05T16:50:24.211", format="isot", scale="utc")
    end_time   = Time("2012-07-25T16:50:24.211", format="isot", scale="utc")
    step_days = 1
    kernel_manager_kwargs = {
        "frame": "MOON_PA_DE440",
        "detailed": True,
        "pre_download_kernels": False,
        "diviner_ck": True,
        "lroc_ck": False,
        "pre_load_static_kernels": True,
        "keep_dynamic_kernels": True,
    }
    filter_type = "lunar_pit"
    filter_kwargs = {"hard_radius": 35}
```

```python
# src/experiments/extraction/lunar_pit_data_extraction_test.py
from astropy.time import Time, TimeDelta
from .base_extraction_experiment import BaseExtractionConfig

class DIVINERTestExtractorConfig(BaseExtractionConfig):
    experiment_name = "lunar_pit_extraction_test"
    instrument_names = ["DIVINER", "LOLA"]
    interval_name = "lunar_pit_run"
    kernel_manager_type = "LRO"

    start_time = Time("2009-07-05T00:00:00.000", format="isot", scale="utc")
    end_time = start_time + TimeDelta(45, format="jd")
    step_days = 1

    kernel_manager_kwargs = {
        "frame": "MOON_PA_DE440",
        "detailed": True,
        "pre_download_kernels": False,
        "diviner_ck": True,
        "lroc_ck": True,
        "pre_load_static_kernels": True,
        "keep_dynamic_kernels": True,
    }

    filter_type = "lunar_pit"
    filter_kwargs = {"hard_radius": 5}
    custom_filter_kwargs = {"MiniRF": {"hard_radius": 5}}
```

In each `__init__.py`, register your configs:

```python
# src/experiments/simulations/__init__.py
from .test_lro_short_simulation import TestLROShortSimulationConfig

# src/experiments/extraction/__init__.py
from .lunar_pit_extraction_test import DIVINERTestExtractorConfig
```

#### Configuration Parameters

- **experiment_name**: unique key used by the CLI to lookup the config.
    
- **instrument_names**: list of instrument IDs to include (e.g., `"DIVINER"`, `"LOLA"`).
    
- **kernel_manager_type**: selects which static/dynamic kernel loader to use (e.g., `"LRO"`).
    
- **start_time**, **end_time**: `astropy.time.Time` instances defining the overall time window.
    
- **step_days**: integer number of days per subtask chunk.
    
- **kernel_manager_kwargs**: passed directly to the kernel manager constructor for frame, CK flags, etc.
    
- **filter_type** & **filter_kwargs**: specify which spatial filter to apply and its parameters.
    
- **custom_filter_kwargs** (extraction only): per-instrument override of `filter_kwargs`.
    

---

### All scripts has to be ran with virtual env activated

### 1 Simulation (Remote-Sensing Pre-Scan)

Invoke a simulation experiment by name:

```bash
python3 run python src/manual_scripts/assign_tasks.py \
  --task remote_sensing \
  --config-name test_lro_short_simulation \
  --name sim_test_01
```

- **Lookup**: searches `BaseSimulationConfig.registry["test_lro_short_simulation"]`.
    
- **Workflow**: searches for timestamps when instruments have an area of interest in their FOV.
    
- **Splitting**: chunks derived from `start_time`, `end_time`, `step_days`.
    
- **Aggregate** example for DIVINER:
    
    ```bash
python3 run python src/manual_scripts/aggregate_simulation_intervals.py \
      --config-name test_lro_short_simulation \
      --instruments DIVINER \
      --threshold 5 \
      --sim-name sim_test_01
    ```
This task does not run in distributed manner and it's goal is to convert timestamps from the simulation loop into continuous time intervals. You can configure the threshold \[Km\] and instrument individually, to yank the most not-wanted data.

---

### 2 Extraction (Pixel-Level Re-Projection)

Run extraction:

```bash
python3 run python src/manual_scripts/assign_tasks.py \
  --task extraction \
  --config-name lunar_pit_extraction_test \
  --name extract_test_01
```

- **Lookup**: via `BaseExtractionConfig.registry["lunar_pit_extraction_test"]`.
    
- **Workflow**: `DataFetchingEngine` loads the intervals generated by aggregation script from online resources, reprojects geometry with our SPICE kernel configuration for each datapoint obtained, when the point falls outside of the filter and it's `hard_radius`, it's eliminated prematurely.
    
- **Aggregation**: Data are aggregated into MongoDB timeseries (Upgrading data aggregation for better suited database system would allow even more scaling)
    

---

### 3 Monitoring

You can monitor Celery worker states with flower `localhost:5555` and the master node utilization of resources with netdata `localhost:19999`.
## SPICE Engine

The SPICE Engine is organized as a fully modular, plugin‑driven framework with two orthogonal layers:

1. **Kernel Managers** load/unload SPICE kernels by time windows.
    
2. **Instruments** consume kernels to compute geometry (position, attitude, FOV).
    

Both layers rely on abstract base classes, a registry/factory pattern for dynamic discovery, and minimal boilerplate for extension.

---

## SPICE Engine

The SPICE Engine underpins all geometric computations by orchestrating SPICE kernel loading and instrument projection in a fully modular way. Its two orthogonal layers—**Kernel Managers** and **Instruments**—are powered by abstract base classes, mixins, and a registry/factory pattern.

---

### Kernel Managers

**Purpose:** manage SPICE kernels (LSK, PCK, FK, IK, CK, SPK, DSK) over time, ensuring only relevant files are loaded to SPICE and memory use remains bounded.

#### BaseKernelManager

Defined in `SPICE/kernel_utils/base_kernel_manager.py`, this abstract class:

- Initializes **static_kernels** (per‐type lists) and an empty **dynamic_kernels** list via its constructor:
    
    ```python
    def __init__(
        self,
        min_required_time=None,
        max_required_time=None,
        pre_load_static_kernels=True,
        frame="MOON_ME",
        detailed=False,
        pre_download_kernels=False,
        keep_dynamic_kernels=True,
        **kwargs,
    ):
        # populate static_kernels with BaseKernel instances
        self.dynamic_kernels = []
        self.add_mixin_kernels(min_required_time=min_required_time,
                                max_required_time=max_required_time,
                                **kwargs)
        if pre_load_static_kernels:
            self.load_static_kernels()
    ```
    
- Provides key methods:
    
    - `load_static_kernels()`: calls `StaticKernelLoader` on `static_kernels` to `spice.furnsh` all static files.
        
    - `step(time: Time) -> bool`: iterates through each entry in `dynamic_kernels`, invoking its `reload_kernels(time)` to load or unload dynamic segments. Returns `True` if all succeed.
        
    - `unload_all()`: unloads every static and dynamic kernel from SPICE.
        
    - `add_mixin_kernels(**kwargs)`: walks the class hierarchy, invoking each mixin’s `setup_kernels(**kwargs)` to register mission‑specific kernels.

**In order to** use the custom defined Instrument in tasks by only referencing it's name in experiment configuration, it's necessary to add it to `KERNEL_MANAGER_MAP` in `src/pipeline/[tasksextractor.py|simulator.py]`.

#### Mixing in Mission Logic

Kernel managers combine `BaseKernelManager` with lightweight mixins in `SPICE/kernel_utils/kernel_manager_mixins/*.py`:

- **Mixins** implement `setup_kernels(self, **kwargs)`, appending `BaseKernel`, `AutoUpdateKernel`, or dynamic‐loader instances to the manager’s pools.
    
- Example: `LROKernelManagerMixin` registers LSK, PCK, FK, IK via `BaseKernel`, and conditionally appends `DynamicKernel` loaders for Diviner/LROC CKs.
    

##### Extension Example

To support a new target (e.g. Mars), define:

```python
from src.SPICE.kernel_utils.base_kernel_manager import BaseKernelManager
from src.SPICE.kernel_utils.kernel_manager_mixins.base_mixin import BaseKernelManagerMixin

class MarsKernelMixin(BaseKernelManagerMixin):
    def setup_kernels(self, **kwargs):
        # append BaseKernel and dynamic loaders for Mars missions
        self.static_kernels['spk'].append(
            BaseKernel(mars_url('spk/mro.bsp'), mars_path('spk/mro.bsp'))
        )
        # ... add more

class MarsKernelManager(MarsKernelMixin, BaseKernelManager):
    pass
```

Register in `SPICE/kernel_utils/kernel_manager_mixins/__init__.py` or import for discovery.

Dynamic kernels (in `dynamic_kernels.py`) implement the `reload_kernels(Time)` interface by parsing metadata labels or filename patterns to determine their valid time range, downloading on-demand, loading into SPICE, and unloading when out of scope. This keeps memory overhead minimal without manual intervention.

#### Kernel Types

| Class                      | Purpose                                                                                              |
| -------------------------- | ---------------------------------------------------------------------------------------------------- |
| **BaseKernel**             | Abstract kernel with resumable download, SPICE load/unload, and cleanup                              |
| **HTTPRangeFile**          | Extends BaseKernel; partial downloads via HTTP Range headers with retry and timeout logic.           |
| **ZipHTTPRangeFile**       | Wraps HTTPRangeFile to stream and extract ZIP-packaged kernels.                                      |
| **AutoUpdateKernel**       | Monitors a URL or directory listing, auto-picks newest kernel matching a regex, and manages versions |
| **LBLKernel**              | Pairs binary kernels with `.LBL` metadata labels, parses validity intervals, and supports updates.   |
| **DynamicKernel**          | Adds time bounds (`time_start`, `time_stop`), implements on-demand segment loading/unloading         |
| **LBLDynamicKernel**       | LBL-aware dynamic kernel using PDS label metadata for segment validity.                              |
| **DSKKernel**              | Loads static DSK files for high-resolution shape models (e.g., local mesh segments).                 |
| **DetailedModelDSKKernel** | Auto-generates detailed LOLA-based DSK meshes at runtime, caches via BunnyCDN for reuse.             |

Each kernel type can be composed via mixins in `kernel_manager_mixins` to create mission- or instrument-specific managers.

---

#### Dynamic Kernel Managers

| Class                    | Purpose                                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **DynamicKernelManager** | Coordinates a collection of `DynamicKernel` instances, invoking `reload_kernels(et)` to load or unload segments based on their time bounds |
| **PriorityKernelLoader** | Wraps multiple kernel managers (e.g., static and dynamic) to enforce priority load/unload order                                            |
| **StaticKernelLoader**   | Internal utility that bulk-registers static kernels (LSK, PCK, FK, IK) via SPICE furnsh.                                                   |

These managers integrate seamlessly with `BaseKernelManager` through mixins and inheritance, enabling complex load strategies (e.g., static-first, dynamic-on-demand) with minimal custom code.
---

### Instruments

**Purpose:** Convert SPICE‐managed spacecraft geometry into instrument‐level projection vectors and footprints.

#### BaseInstrument

Defined in `src/SPICE/instruments/instrument.py`.

- **Attributes**:
    
    - `name`: unique instrument identifier matching NAIF or factory key.
        
    - `frame`: SPICE reference frame for boresight transforms.
        
    - `satellite_name`: NAIF name for SPK/CK lookups.
        
- **Property**:
    
    - `sub_instruments` (List[SubInstrument]): individual sensor elements.
        
- **Core Methods**:
    
    ```python
    def boresight(self, et: float) -> np.ndarray:
        """Compute and return the average boresight vector across sub_instruments."""
    
    def bounds(self, et: float) -> List[np.ndarray]:
        """Return extreme FOV boundary vectors from all sub_instruments."""
    
    def project_vector(self, et: float, vector: np.ndarray) -> ProjectionPoint:
        """Intersect a look vector with the planetary surface via `spice.sincpt` or DSK tracing."""
    
    def project_boresight(self, et: float) -> ProjectionPoint:
        """Shortcut for projecting the boresight vector."""
    
    def project_bounds(self, et: float) -> List[ProjectionPoint]:
        """Shortcut for projecting all boundary vectors."""
    ```
    

#### SubInstrument Hierarchy

Located in `src/SPICE/instruments/subinstruments.py`. Each implements pixel/beam-level geometry.


- **SubInstrument**
	- Takes NAIF_ID as the only constructor parameter, loading all information from instrument SPICE kernel.
- **ImplicitSubInstrument**
    - Manually defined unit vector(s) when SPICE FOV data is unavailable.
- **DivinerSubInstrument**
    - Represents one of 9 Diviner channels, each spawning 21 implicit pixel vectors.
    - Reads boresight and boundary directions from SPICE kernel variables (`INS-85205_*`).
- **MiniRFSubInstrument**
    - Models fan-beam radar by generating edge rays based on documented half-angle parameters.

#### Extension Example

Define and register a custom instrument:

```python
# src/SPICE/instruments/custom_sensor.py
from src.SPICE.instruments.instrument import BaseInstrument
from src.SPICE.instruments.subinstruments import ImplicitSubInstrument

class CustomSensor(BaseInstrument):
    name = "CustomSensor"
    frame = "CUSTOM_FRAME"
    satellite_name = "CUSTOM_SAT"

    @property
    def sub_instruments(self):
        # Single-pixel sensor pointing along X-axis
        return [ImplicitSubInstrument(naif_id=4000, vector=[1,0,0])]
```


In order to use the custom defined Instrument in tasks by only referencing it's name in experiment configuration, it's necessary to add it to `INSTRUMENT_MAP` in `src/pipeline/[tasksextractor.py|simulator.py]`.

### Data Connectors

`DataConnector` classes bridge raw mission archives and the extraction engine by loading, parsing, and slicing time‐tagged records. Each connector subclasses `BaseDataConnector` (`src/data_fetchers/base_connector.py`) and must define the following interface:

```python
class BaseDataConnector(ABC):
    # Class attributes to set in subclasses
    name: str                 # unique key
    orbiting_body: str        # e.g., "MOON"
    timeseries: Any           # timeseries definition for MongoDB
    indices: Any              # list of timeseries "columns" to index

    def __init__(self, time_intervals: IntervalList, kernel_manager: BaseKernelManager):
        """Discover files, prefetch first, and parse initial data."""

    @abstractmethod
    def discover_files(self, time_intervals: IntervalList) -> List[VirtualFile]:
        """Return VirtualFile list covering the specified time intervals."""

    @abstractmethod
    def _parse_current_file(self) -> None:
        """Parse bytes from `current_file.buffer` into an internal table or array."""

    @abstractmethod
    def _get_interval_data_from_current_file(
        self, time_interval: TimeInterval, instrument: BaseInstrument, filter_obj: BaseFilter
    ) -> List[Dict]:
        """Slice parsed data for the given interval, instrument, and filter."""

    @abstractmethod
    def process_data_entry(
        self, data_entry: Dict, instrument: BaseInstrument, filter_obj: BaseFilter
    ) -> Dict:
        """Project a single raw entry via instrument & filter, returning a final record."""

    def read_interval(
        self, time_interval: TimeInterval, instrument: BaseInstrument, filter_obj: BaseFilter
    ) -> List[Dict]:
        """High-level loop: loads all overlapping files, parses slices, and aggregates records."""
```

Concrete connectors included:

- **DivinerDataConnector**
    
    - **Data**: timestamped brightness temperatures for nine channels (PDS3 `.TAB` in ZIP).
        
- **LOLADataConnector**
    
    - **Data**: elevation profiles (`.DAT` binary), with lat/lon, time, and incidence angle.
        
- **MiniRFDataConnector**
    
    - **Data**: synthetic aperture radar swaths (custom binary), outputting backscatter arrays with geo/time tags.
        

To extend:

1. Create a subclass of `BaseDataConnector`, set `name`, `orbiting_body`, `timeseries`, and `indices`.
    
2. Implement the four abstract methods above.
    
3. Add your class to `DATA_CONNECTOR_MAP` in `src/data_fetchers/data_extractor.py`.
    

Once registered, the extraction engine will automatically use your connector when referenced in an experiment configuration.
### Abstractions


| Abstraction       | Purpose                                                          |
| ----------------- | ---------------------------------------------------------------- |
| `TimeInterval`    | `[start_et, end_et]` w/ validation & overlap ops                 |
| `IntervalManager` | merges per‑instrument lists, slices interval lists into subtasks |
| `VirtualFile`     | resumable async download → RAM/ROM buffer                        |

#### Example: VirtualFile

```python
from src.structures import TimeInterval, VirtualFile
iv = TimeInterval(et0, et1)
file = VirtualFile(url, iv)
file.download()
bytes_io = file.buffer  # ready for numpy / pandas
```

More structures are implemented in `src/structures` and more information is in the code itself, particularly docstrings.



---
## Analysis

All analysis mentioned in the thesis is in one of the Jupyter notebooks in `/notebooks`. Install modules not used in the main workflow with `poetry add PACKAGE_NAME` or `pip3 install PACKAGE_NAME`, or directly within the notebook - `!pip3 install PACKAGE_NAME`.
