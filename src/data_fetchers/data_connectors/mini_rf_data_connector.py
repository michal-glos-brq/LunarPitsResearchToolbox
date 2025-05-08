import os
import pvl
import urllib
import logging
import pickle
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional
from time import sleep
from itertools import product, chain


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
from src.global_config import SUPRESS_TQDM, TQDM_NCOLS, HDD_BASE_PATH
from src.filters import BaseFilter
from src.SPICE.instruments.instrument import BaseInstrument
from src.SPICE.kernel_utils.kernel_management import BaseKernelManager


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


minirf_url = lambda url: urllib.parse.urljoin(MINIRF_BASE_URL, url)
pickle_file = os.path.join(Path(__file__).resolve().parent, MINNIRF_REMOTE_CACHE_FILE)
pickle_file_lock = os.path.join(HDD_BASE_PATH, MINNIRF_REMOTE_CACHE_FILE + ".lock")

SUBDIVISION_MIN_SIDE_LEN = 4


class MiniRFDataConnector(BaseDataConnector):

    name = "MiniRF"

    timeseries = {
        "timeField": "timestamp",
        "metaField": "meta",
        "granularity": "seconds",
    }

    indices = ["meta.et", "meta.extraction_name", "cx_projected", "cy_projected", "cz_projected"]

    def __init__(
        self,
        time_intervals: IntervalList,
        kernel_manager: BaseKernelManager,
    ):
        super().__init__(time_intervals, kernel_manager)
        self.cache = None

    @property
    def dataset_structure(self):
        """
        This method fetches and parses dataset metadata. The pickled file is provided with the repository, because PDS servers
        do not like sending thousands of 5 Kb files at once. If an updated version is required, delete the pickle file and pubslish a new one
        """
        # If for some reason the pickle file is not present, it's possible worker will not have permissions to write it,
        # hence running it locally, pushing to git and updating is the way to update this data structure, though the data product is dead now
        with FileLock(pickle_file_lock, timeout=-1, poll_interval=1):
            if os.path.isfile(pickle_file):
                with open(pickle_file, "rb") as f:
                    data_structure = pickle.load(f)
                logger.info(f"Loaded data structure from {pickle_file}")
                # Just make sure it's sorted
                data_structure = sorted(data_structure, key=lambda x: TimeInterval(*x["interval"]))
                return data_structure

            logger.warning(
                f"Pickle file {pickle_file} not found. Fetching data from PDS servers. This might take an hour, easily even more ..."
            )

            import requests_cache

            requests_cache.install_cache(
                cache_name="mini_rf_dataset_cache",
                backend="sqlite",  # SQLite can throw errors when accessed from multiple threads
                expire_after=3600 * 12,  # 12 hours, well why not, sometimes the servers are quite hostile
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

            for url_suffix in tqdm(
                remote_data_structure,
                desc="Parsind PD3 labels ...",
                ncols=TQDM_NCOLS,
            ):
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

        logging.info(f"Found {sum(dataset_structure_mask)} MiniRF data files intersecting with our time intervals")

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
        line_pixels = np.arange(0, self.current_file.metadata["line_samples"])

        arr = arr.reshape((self.current_file.metadata["lines"], self.current_file.metadata["line_samples"], 4))

        self.current_file.data = arr
        self.current_file.timestamps = timestamps
        self.current_file.line_pixels = line_pixels

    def _project_pixel(self, instrument, sub_instrument, et, line, lines):
        if self.cache is not None and (et, line, lines) in self.cache:
            return self.cache[(et, line, lines)]
        self.kernel_manager.step_et(et)
        pixel_vector = sub_instrument.look_vector_for_pixel(line, lines)
        projection_vector = sub_instrument.transform_vector(instrument.frame, pixel_vector, et=et)
        return_value = instrument.project_vector(et, projection_vector)
        if self.cache is not None:
            self.cache[(et, line, lines)] = return_value
        return return_value

    def _get_subinstrument_from_mode(self, subinstrument, mode):
        if mode in MINI_RF_S_MODES_IDX:
            return subinstrument.s
        elif mode in MINI_RF_X_MODES_IDX:
            return subinstrument.x
        else:
            raise ValueError(f"Unknown mode {mode}")

    def _get_interval_data_from_current_file(
        self, time_interval: TimeInterval, instrument: BaseInstrument, filter_obj: BaseFilter
    ) -> List[Dict]:
        """Obtain data for the given time interval from the current file. To avoid returning too much data, prefilter the data with preevaluation metrics"""
        if self.current_file is None:
            return []

        timestamp_mask = (self.current_file.timestamps > time_interval.start_et) & (
            self.current_file.timestamps < time_interval.end_et
        )
        data_slice_len = timestamp_mask.sum()
        if data_slice_len < 2:
            logger.warning(f"Insuffiient data for interval {time_interval}.")
            return []

        # Numpy array with data slice
        data_slice = self.current_file.data[timestamp_mask, :, :]
        # Numpy array with timestamps (et) slice
        t = self.current_file.timestamps[timestamp_mask]
        # Line indices (not a slice, we take all lines (same et for all pixels of line in this dim.))
        l = self.current_file.line_pixels
        # Init cache
        self.cache = {}

        mode = MINI_RF_MODES[self.current_file.metadata["instrument_mode_id"]]
        sub_instrument = self._get_subinstrument_from_mode(instrument, mode)
        # How wide a single radar line is (int, number of pixels)
        lines = self.current_file.metadata["line_samples"]

        points_to_project = [
            (t[0], 0, lines),
            (t[0], 1, lines),
            (t[1], 0, lines),
        ]

        projected_points = [self._project_pixel(instrument, sub_instrument, *point) for point in points_to_project]
        timestemp_delta = np.linalg.norm(projected_points[0].projection - projected_points[2].projection)
        lines_delta = np.linalg.norm(projected_points[0].projection - projected_points[1].projection)

        max_step_t = int(filter_obj.hard_radius / timestemp_delta)
        max_step_l = int(filter_obj.hard_radius / lines_delta)
        max_step = min(max_step_t, max_step_l)
        min_step = SUBDIVISION_MIN_SIDE_LEN



        # def slice_data(start_id_t, stop_id_t, start_id_l, stop_id_l) -> List[Dict]:
        #     df = pd.DataFrame(
        #         {
        #             "et": t[start_id_t:stop_id_t].repeat(stop_id_l - start_id_l),
        #             "mode": mode,
        #             "line": np.tile(l[start_id_l:stop_id_l], stop_id_t - start_id_t),
        #             "lines": lines,
        #             "ch1": data_slice[start_id_t:stop_id_t, start_id_l:stop_id_l, 0].flatten(),
        #             "ch2": data_slice[start_id_t:stop_id_t, start_id_l:stop_id_l, 1].flatten(),
        #             "ch3": data_slice[start_id_t:stop_id_t, start_id_l:stop_id_l, 2].flatten(),
        #             "ch4": data_slice[start_id_t:stop_id_t, start_id_l:stop_id_l, 3].flatten(),
        #         }
        #     )
        #     mask = (df[["ch1", "ch2", "ch3", "ch4"]] == 0).all(axis=1)
        #     return df[~mask].reset_index(drop=True).to_dict(orient="records")
        
        def slice_data(start_id_t, stop_id_t, start_id_l, stop_id_l) -> List[Dict]:
            # pull out the block
            block = data_slice[start_id_t:stop_id_t, start_id_l:stop_id_l, :]  # shape (T_block, L_block, 4)
            # mask of non-zero pixels across all 4 channels
            nz_mask = ~np.all(block == 0, axis=2)                               # shape (T_block, L_block)
            if not nz_mask.any():
                return []  # nothing to do here

            # find the (i,j) subscripts of valid pixels
            tt, ll = np.nonzero(nz_mask)                                        # both shape (N,)
            # corresponding ETs and global line-indices
            et_vals   = t[start_id_t:stop_id_t][tt]                             # shape (N,)
            line_vals = l[start_id_l:stop_id_l][ll]                             # shape (N,)
            chans     = block[tt, ll, :]                                        # shape (N,4)

            # build your list of dicts
            recs = []
            for et_i, line_i, (c1, c2, c3, c4) in zip(et_vals, line_vals, chans):
                recs.append({
                    "et":    float(et_i),
                    "mode":  mode,
                    "line":  int(line_i),
                    "lines": lines,
                    "ch1":   float(c1),
                    "ch2":   float(c2),
                    "ch3":   float(c3),
                    "ch4":   float(c4),
                })
            return recs

        def cut_and_filter(start_id_t, stop_id_t, start_id_l, stop_id_l, init: bool = False) -> List[Dict]:
            # If this is hit immediately upon starting, it can be outside our interest
            # but it will eventually get filtered out and in this scale, it is irrelevant
            if stop_id_t - start_id_t <= min_step or stop_id_l - start_id_l <= min_step:
                if not init:
                    return slice_data(start_id_t, stop_id_t, start_id_l, stop_id_l)
                
                ets_edges = (t[start_id_t], t[stop_id_t - 1])
                lines_edges = (l[start_id_l], l[stop_id_l - 1])
                projected_points = [
                    self._project_pixel(instrument, sub_instrument, et, line, lines) for et, line in product(ets_edges, lines_edges)
                ]
                # It's small, if even one projection hits within, we take it all ...
                for et, line in product(ets_edges, lines_edges):
                    pixel_projection = self._project_pixel(instrument, sub_instrument, et, line, lines).projection
                    if filter_obj.rank_point(pixel_projection) <= filter_obj.hard_radius:
                        return slice_data(start_id_t, stop_id_t, start_id_l, stop_id_l)
                return []

            center_id_t = (start_id_t + stop_id_t) // 2
            center_id_l = (start_id_l + stop_id_l) // 2

            ets_edges = (t[start_id_t], t[center_id_t], t[stop_id_t - 1])
            lines_edges = (l[start_id_l], l[center_id_l], l[stop_id_l - 1])
            projected_points = [
                self._project_pixel(instrument, sub_instrument, et, line, lines) for et, line in product(ets_edges, lines_edges)
            ]
            within_filter_radius = np.array(
                [filter_obj.rank_point(point.projection) < filter_obj.hard_radius for point in projected_points]
            ).reshape((3, 3))

            # All corners and center are within filter? Slice all data
            if within_filter_radius.all():
                return slice_data(start_id_t, stop_id_t, start_id_l, stop_id_l)
            # None of corners and neither center is within filter? Empty list
            elif (~within_filter_radius).all():
                return []
            # If none of above is true, we have to solve each subdivision on it's own
            else:
                slicing_arguments = []
                further_cutting_arguments = []

                # If all corners of particular subdivision square are within, the whole slice is within
                if within_filter_radius[:2, :2].all():
                    slicing_arguments.append((start_id_t, center_id_t, start_id_l, center_id_l))
                if within_filter_radius[:2, 1:].all():
                    slicing_arguments.append((start_id_t, center_id_t, center_id_l, stop_id_l))
                if within_filter_radius[1:, :2].all():
                    slicing_arguments.append((center_id_t, stop_id_t, start_id_l, center_id_l))
                if within_filter_radius[1:, 1:].all():
                    slicing_arguments.append((center_id_t, stop_id_t, center_id_l, stop_id_l))
                # If any of the corners of the subdivision square are within, further slicing is required
                if within_filter_radius[:2, :2].any():
                    further_cutting_arguments.append((start_id_t, center_id_t, start_id_l, center_id_l))
                if within_filter_radius[:2, 1:].any():
                    further_cutting_arguments.append((start_id_t, center_id_t, center_id_l, stop_id_l))
                if within_filter_radius[1:, :2].any():
                    further_cutting_arguments.append((center_id_t, stop_id_t, start_id_l, center_id_l))
                if within_filter_radius[1:, 1:].any():
                    further_cutting_arguments.append((center_id_t, stop_id_t, center_id_l, stop_id_l))

                return_value = [slice_data(*args) for args in slicing_arguments] 
                return_value += [cut_and_filter(*args) for args in further_cutting_arguments]
                return list(chain.from_iterable(return_value))

        t_args_values = np.append(np.arange(0, len(t), max_step), len(t))
        l_args_values = np.append(np.arange(0, len(l), max_step), len(l))
        t_args = np.array([t_args_values[:-1], t_args_values[1:]]).T
        l_args = np.array([l_args_values[:-1], l_args_values[1:]]).T
        args = np.array(list(product(t_args, l_args))).reshape((-1, 4))

        return_value_list = []
        for arg in args:
            return_value_list.append(cut_and_filter(*arg, init=True))

        self.cache = None
        return list(chain.from_iterable(return_value_list))


    def process_data_entry(
        self, data_entry: Dict, instrument: BaseInstrument, filter_obj: BaseFilter
    ) -> Optional[Dict]:
        sub_instrument = self._get_subinstrument_from_mode(instrument, data_entry["mode"])
        projection: ProjectionPoint = self._project_pixel(
            instrument, sub_instrument, data_entry["et"], data_entry["line"], data_entry["lines"]
        )

        if filter_obj.point_pass(projection.projection):
            data_entry.update(projection.to_data())
            return data_entry
        else:
            return None
