import os
import pvl
import urllib
import logging
import pickle
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional
from time import sleep
from tqdm import tqdm

from filelock import FileLock
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup as bs

from src.structures import VirtualFile, TimeInterval, IntervalList, ProjectionPoint
from src.data_fetchers.data_connectors.base_data_connector import BaseDataConnector
from src.data_fetchers.config import (
    MINNIRF_REMOTE_CACHE_FILE,
    MINIRF_BASE_URL,
    MINI_RF_URLS,
    MINI_RF_MODES,
    MINI_RF_S_MODES_IDX,
    MINI_RF_X_MODES_IDX,
)
from src.SPICE.utils import DatetimeToETDecoder
from src.global_config import SUPRESS_TQDM, TQDM_NCOLS
from src.filters import BaseFilter
from src.SPICE.instruments.instrument import BaseInstrument


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


minirf_url = lambda url: urllib.parse.urljoin(MINIRF_BASE_URL, url)
pickle_file = os.path.join(Path(__file__).resolve().parent, MINNIRF_REMOTE_CACHE_FILE)


class MiniRFDataConnector(BaseDataConnector):

    name = "MiniRF"

    timeseries = {
        "timeField": "timestamp",
        "metaField": "meta",
        "granularity": "seconds",
    }

    indices = ["meta.et", "meta.extraction_name", "cx_projected", "cy_projected", "cz_projected"]

    @property
    def dataset_structure(self):
        """
        This method fetches and parses dataset metadata. The pickled file is provided with the repository, because PDS servers
        do not like sending thousands of 5 Kb files at once. If an updated version is required, delete the pickle file and pubslish a new one
        """
        with FileLock(pickle_file + ".lock", timeout=-1, poll_interval=1):
            if os.path.isfile(pickle_file):
                with open(pickle_file, "rb") as f:
                    data_structure = pickle.load(f)
                logger.info(f"Loaded data structure from {pickle_file}")
                # Just make sure it's sorted
                data_structure = sorted(data_structure, key=lambda x: TimeInterval(*x["interval"]))
                return data_structure

            logger.warning(f"Pickle file {pickle_file} not found. Fetching data from PDS servers. This might take an hour, easily even more ...")

            import requests_cache

            requests_cache.install_cache(
                cache_name="mini_rf_dataset_cache",
                backend="sqlite",
                expire_after=3600 * 12,  # 12 hours
            )

            # Fetch data addresses
            while True:
                try:
                    responses = [BaseDataConnector.fetch_url(minirf_url(url_suffix)) for url_suffix in MINI_RF_URLS]
                    soups = [bs(resp.text, "html.parser") for resp in responses]
                    hrefs = [soup.find_all("a") for soup in soups]
                    links = [
                        href.get("href")
                        for (url_suffix, _hrefs) in zip(MINI_RF_URLS, hrefs)
                        for href in _hrefs
                        if href.get("href").startswith(url_suffix) and not href.get("href").endswith("mosaics/")
                    ]
                    responses = [
                        BaseDataConnector.fetch_url(minirf_url(os.path.join(url_suffix, "level1/")), timeout=30)
                        for url_suffix in tqdm(links)
                    ]
                    texts = [res.text for res in responses]
                    soups = [bs(text, "html.parser") for text in texts]
                    hrefs = [href for soup in soups for href in soup.find_all("a")]
                    links = [href.get("href") for href in hrefs if "level1" in href.get("href")]
                    break
                except Exception as e:
                    logging.error(f"Error fetching data: {e}")
                    sleep(60)
                    continue

            remote_data_structure = defaultdict(dict)
            for link in links:
                path, suffix = link.rsplit(".", 1)
                remote_data_structure[path][suffix] = link

            # Fetch the lbl files
            while True:
                try:
                    _remote_data_structure_keys = []
                    for url_suffix in remote_data_structure:
                        if not remote_data_structure[url_suffix].get("lbl_text"):
                            _remote_data_structure_keys.append(url_suffix)

                    with tqdm(
                        total=len(_remote_data_structure_keys),
                        desc="Downloading Mini RF data",
                        disable=SUPRESS_TQDM,
                        ncols=TQDM_NCOLS,
                    ) as pbar:
                        for url_suffix in _remote_data_structure_keys:
                            remote_data_structure[url_suffix]["lbl_text"] = self.fetch_url(
                                minirf_url(remote_data_structure[url_suffix]["lbl"]), timeout=60
                            ).text
                            pbar.update(1)
                    break
                except Exception as e:
                    logging.error(f"Error fetching data: {e}")
                    sleep(500)
                    continue

            for url_suffix in tqdm(remote_data_structure, desc="Parsind PD3 labels ...", ncols=TQDM_NCOLS, ):
                remote_data_structure[url_suffix]["meta"] = pvl.loads(
                    remote_data_structure[url_suffix]["lbl_text"], decoder=DatetimeToETDecoder()
                )

            # Turn it off if we got the data
            requests_cache.uninstall_cache()

            data_structure = []
            for url_suffix in tqdm(remote_data_structure):
                interval = (
                    remote_data_structure[url_suffix]["meta"]["START_TIME"],
                    remote_data_structure[url_suffix]["meta"]["STOP_TIME"],
                )
                data_link = minirf_url(remote_data_structure[url_suffix]["img"])
                coni_link = minirf_url(remote_data_structure[url_suffix]["txt"])
                metadata = {
                    "records": remote_data_structure[url_suffix]["meta"]["FILE_RECORDS"],
                    "bands": remote_data_structure[url_suffix]["meta"]["IMAGE"]["BANDS"],
                    "band_names": remote_data_structure[url_suffix]["meta"]["IMAGE"]["BAND_NAME"],
                    "lines": remote_data_structure[url_suffix]["meta"]["IMAGE"]["LINES"],
                    "line_exposure_duration": remote_data_structure[url_suffix]["meta"]["IMAGE"][
                        "LINE_EXPOSURE_DURATION"
                    ],
                    "line_samples": remote_data_structure[url_suffix]["meta"]["IMAGE"]["LINE_SAMPLES"],
                    "instrument_mode_id": remote_data_structure[url_suffix]["meta"]["INSTRUMENT_MODE_ID"],
                    "central_frequency_GHz": remote_data_structure[url_suffix]["meta"]["CENTER_FREQUENCY"].value,
                }
                data_structure.append(
                    {"interval": interval, "data_link": data_link, "metadata": metadata, "coni": coni_link}
                )

            data_structure = sorted(data_structure, key=lambda x: TimeInterval(*x["interval"]))
            # Save the data structure to a pickle file
            with open(pickle_file, "wb") as f:
                pickle.dump(data_structure, f)
            logger.info(f"Saved data structure to {pickle_file}")
            return data_structure

    def discover_files(self, time_intervals: IntervalList):
        dataset_structure = self.dataset_structure
        dataset_data_intervals = IntervalList([struct["interval"] for struct in dataset_structure])
        dataset_structure_mask = dataset_data_intervals.intersection_mask(time_intervals)

        logging.info("Found %s MiniRF data files intersecting with our time intervals".format(sum(dataset_structure_mask)))

        virtual_files: List[VirtualFile] = []
        for file_dict, mask in zip(dataset_structure, dataset_structure_mask):
            if mask:
                interval = TimeInterval(file_dict["interval"][0], file_dict["interval"][1])
                virtual_files.append(VirtualFile(file_dict["data_link"], interval, file_dict["metadata"]))

        # Sort the files by their time intervals
        virtual_files.sort(key=lambda x: x.interval)
        return virtual_files

    def _parse_current_file(self):
        self.current_file.wait_to_be_downloaded()
        # Load the raster image into RAM
        raw = self.current_file.file.read()
        arr = np.frombuffer(raw, dtype=np.float32)

        # Generate timestamps for the image lines
        timestamps = (
            self.current_file.interval.start_et
            + np.arange(0, self.current_file.metadata["lines"]) * self.current_file.metadata["line_exposure_duration"]
        )
        # Multiply them to create pandas dataframe
        timestamps = timestamps.repeat(self.current_file.metadata["line_samples"])
        line_pixels = np.arange(0, self.current_file.metadata["line_samples"])
        line_pixels = np.tile(line_pixels, self.current_file.metadata["lines"])

        arr = arr.reshape((self.current_file.metadata["lines"], self.current_file.metadata["line_samples"], 4))

        df = pd.DataFrame(
            {
                "et": timestamps,
                "mode": MINI_RF_MODES[self.current_file.metadata["instrument_mode_id"]],
                "line": line_pixels,
                "lines": self.current_file.metadata["line_samples"],
                "ch1": arr[:, :, 0].flatten(),
                "ch2": arr[:, :, 1].flatten(),
                "ch3": arr[:, :, 2].flatten(),
                "ch4": arr[:, :, 3].flatten(),
            }
        )

        mask = (df[["ch1", "ch2", "ch3", "ch4"]] == 0).all(axis=1)
        df = df[~mask].reset_index(drop=True)
        self.current_file.data = df

    def _get_interval_data_from_current_file(self, time_interval: TimeInterval) -> List[Dict]:
        if self.current_file is None:
            return []
        data = self.current_file.data.loc[
            (self.current_file.data.et > time_interval.start_et) & (self.current_file.data.et < time_interval.end_et)
        ].copy()
        return data.to_dict(orient="records")

    def process_data_entry(
        self, data_entry: Dict, instrument: BaseInstrument, filter_obj: BaseFilter
    ) -> Optional[Dict]:
        if data_entry["mode"] in MINI_RF_S_MODES_IDX:
            sub_instrument = instrument.s
        elif data_entry["mode"] in MINI_RF_X_MODES_IDX:
            sub_instrument = instrument.x
        else:
            logger.warning(f"Unknown mode {data_entry['mode']}")
            return None

        et = data_entry["et"]
        self.kernel_manager.step_et(et)

        pixel_vector = sub_instrument.look_vector_for_pixel(data_entry["line"], data_entry["lines"])
        projection_vector = sub_instrument.transform_vector(instrument.frame, pixel_vector, et=et)

        projection: ProjectionPoint = instrument.project_vector(et, projection_vector)
        if filter_obj.point_pass(projection.projection):
            data_entry.update(projection.to_data())
            return data_entry
        else:
            return None
