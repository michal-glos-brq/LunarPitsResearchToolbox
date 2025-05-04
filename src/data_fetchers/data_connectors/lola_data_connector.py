import os
import pvl
import urllib
import logging
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import Future


from filelock import FileLock
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
import spiceypy as spice
from astropy.time import Time, TimeDelta

from src.structures import VirtualFile, TimeInterval, IntervalList, ProjectionPoint
from src.data_fetchers.data_connectors.base_data_connector import BaseDataConnector
from src.data_fetchers.config import (
    LOLA_BASE_URL,
    LOLA_LBL_FILE_DUMP,
    LOLA_REMOTE_CACHE_FILE,
    LOLA_DATASET_ROOT,
    LOLA_COL_SPECS,
)
from src.SPICE.utils import DatetimeToETDecoder
from src.global_config import SUPRESS_TQDM, TQDM_NCOLS
from src.filters import BaseFilter
from src.SPICE.instruments.instrument import BaseInstrument


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Basically the end of active measurements. Passive values still present, could be potentially processed,
# hence no hard limit on latest_active_lola_date.
latest_active_lola_date = Time("2013-07-01T00:00:00", format="isot", scale="utc")
one_minute_delta = TimeDelta(60, format="sec")
pickle_file = os.path.join(LOLA_LBL_FILE_DUMP, LOLA_REMOTE_CACHE_FILE)
compose_lola_url = lambda path: urllib.parse.urljoin(LOLA_BASE_URL, path)


def lola_rdr_dtype():
    base_map = {
        1: (np.int8, np.uint8),
        2: (np.int16, np.uint16),
        4: (np.int32, np.uint32),
    }
    dt = []
    for spec in LOLA_COL_SPECS:
        name, b, signed, *rest = spec
        count = rest[0] if rest else 1
        if b not in base_map:
            raise ValueError(f"Invalid byte-width {b}")
        base = base_map[b][not signed]
        if count == 1:
            dt.append((name, base))
        else:
            for i in range(count):
                dt.append((f"{name}_{i}", base))
    return np.dtype(dt)


class LOLADataConnector(BaseDataConnector):

    name = "LOLA"

    timeseries = {
        "timeField": "timestamp",
        "metaField": "meta",
        "granularity": "seconds",
    }

    indices = ["meta.et", "meta.extraction_name", "cx_projected", "cy_projected", "cz_projected"]

    @property
    def dataset_structure(self):
        """
        Returns all files mapped to inclusive time intervals they cover

        returns sorted list of tuples (time_interval, file_dict)
        time intervals are Tuples of (start_et, end_et)
        file_dict is a dict with keys as file suffixes (lbl, xml, dat) and values as the url suffixes (without BASE URL)
        """
        with FileLock(pickle_file + ".lock", timeout=-1, poll_interval=1):
            if os.path.isfile(pickle_file):
                with open(pickle_file, "rb") as f:
                    parsed_data_structure = pickle.load(f)
                logger.info(f"Loaded data structure from {pickle_file}")
                return parsed_data_structure

            logger.info(f"Fetching LOLA data structure from {LOLA_BASE_URL}...")

            try:
                resp = BaseDataConnector.fetch_url(compose_lola_url(LOLA_DATASET_ROOT))
                soup = bs(resp.text, "html.parser")
            except Exception as e:
                logger.error(f"Failed to fetch LOLA data structure: {e}")
                raise
            hrefs = soup.find_all("a", href=True)
            links = [href.get("href") for href in hrefs if href.get("href").startswith(LOLA_DATASET_ROOT)]

            tasks = []
            for url_suffix in links:
                url = compose_lola_url(url_suffix)
                tasks.append((url_suffix, BaseDataConnector.fetch_url_async(url)))

            # This will be a dict, where keys are astropy time object, values are dicts, with keys as file suffixe (lbl, xml, dat)
            # and values as the url suffixes (without BASE URL)
            logger.info(f"Walking through the dataset ({len(tasks)}) ... might take several minutes")
            data_structure = defaultdict(dict)
            for i, (url_suffix, task) in enumerate(tasks):
                try:
                    soup = bs(task.result().text, "html.parser")
                except Exception as e:
                    logger.error(f"Failed to fetch LOLA data structure for {url_suffix}: {e}")
                    continue
                hrefs = soup.find_all("a", href=True)
                links = [href.get("href") for href in hrefs if href.get("href").startswith(url_suffix)]
                for _url_suffix in links:
                    path_without_suffix, file_format = _url_suffix.rsplit(".", 1)
                    timestamp_string = path_without_suffix.rsplit("_", 1)[-1]
                    year = 2000 + int(timestamp_string[:2])  # 2009
                    doy = int(timestamp_string[2:5])  # day 194
                    hour = int(timestamp_string[5:7])  # 01
                    minute = int(timestamp_string[7:9])  # 57
                    time_str = f"{year}:{doy:03d}:{hour:02d}:{minute:02d}:00"
                    start_astropy_time = Time(time_str, format="yday", scale="utc")
                    data_structure[start_astropy_time][file_format] = _url_suffix
                if i % 1000 == 0:
                    logger.info(f"Processed {i}/{len(tasks)}")

            logger.info("Finished walking through the dataset")
            sorted_list = sorted(list(data_structure.items()), key=lambda x: x[0])
            timestamps = [key_value[0] for key_value in sorted_list]
            logger.info("Converting timestamps to ephemeris time")
            ets = [spice.utc2et(time.iso) for time in timestamps]
            time_intervals = [(time1, time2 + 60) for time1, time2 in zip(ets[:-1], ets[1:])]
            file_dicts = [key_value[1] for key_value in sorted_list]
            parsed_data_structure = [
                (time_interval, file_dict) for time_interval, file_dict in zip(time_intervals, file_dicts)
            ]

            with open(pickle_file, "wb") as f:
                logger.info(f"Saving data structure to {pickle_file}")
                pickle.dump(parsed_data_structure, f)
                logger.info(f"Saved data structure to {pickle_file}")
            return parsed_data_structure

    def discover_files(self, time_intervals: IntervalList):
        """Discover datafiles from reqeusted time intervals"""
        # Be inclusive, we get precise data from the .lbl file
        inclusive_dataset_structure = [TimeInterval(ds[0] - 60, ds[1] + 60) for ds, _ in self.dataset_structure]

        dataset_slice = []
        for time_interval, (_, file_dict) in zip(inclusive_dataset_structure, self.dataset_structure):
            if time_intervals.get_intervals_intersection(time_interval):
                dataset_slice.append(file_dict)

        lbl_tasks: List[Tuple[Dict[str, str], Future]] = []
        for file_dict in dataset_slice:
            lbl_tasks.append((file_dict, self.fetch_url_async(compose_lola_url(file_dict["lbl"]))))

        virtual_files: List[VirtualFile] = []
        for file_dict, fut in tqdm(lbl_tasks, desc="Downloading metadata", disable=SUPRESS_TQDM, ncols=TQDM_NCOLS):
            resp = fut.result()
            meta = pvl.loads(resp.text, decoder=DatetimeToETDecoder())
            interval = TimeInterval(meta["START_TIME"], meta["STOP_TIME"])
            virtual_files.append(VirtualFile(compose_lola_url(file_dict["dat"]), interval, meta))

        virtual_files.sort(key=lambda x: x.interval)
        remote_interval_list = IntervalList([file.interval for file in virtual_files])
        mask = remote_interval_list.intersection_mask(time_intervals)
        virtual_files = [vf for keep, vf in zip(mask, virtual_files) if keep]
        return virtual_files

    def _parse_current_file(self):
        self.current_file.wait_to_be_downloaded()
        raw = self.current_file.file.read()
        arr = np.frombuffer(raw, dtype=lola_rdr_dtype())
        df = pd.DataFrame(arr)

        df.replace(
            {
                np.int32(-2147483648): np.nan,
                np.uint32(4294967295): np.nan,
                np.uint16(65535): np.nan,
            },
            inplace=True,
        )

        df["et"] = df["TRANSMIT_TIME_0"].astype("float64") + df["TRANSMIT_TIME_1"] * (2**-32)
        df.drop(
            columns=[
                "TRANSMIT_TIME_0",
                "TRANSMIT_TIME_1",
                "MET_SECONDS",
                "SUBSECONDS",
                "EARTH_RANGE",
                "EARTH_PULSE",
                "EARTH_ENERGY",
            ],
            inplace=True,
        )

        stubs = [
            "LONGITUDE",
            "LATITUDE",
            "RADIUS",
            "RANGE",
            "PULSE",
            "ENERGY",
            "BACKGROUND",
            "THRESHOLD",
            "GAIN",
            "SHOT_FLAG",
        ]

        df = df.reset_index(drop=True).rename_axis("row_id").reset_index()
        df = pd.wide_to_long(
            df,
            stubnames=stubs,
            i="row_id",  # identifies each original frame
            j="spot",  # new column will be called “spot”
            sep="_",  # stubs are separated from numbers by “_”
            suffix="\\d+",  # the suffix is one or more digits
        ).reset_index()

        df.drop(columns=["row_id"], inplace=True)
        self.current_file.data = df

    def _get_interval_data_from_current_file(self, time_interval: TimeInterval):
        data = self.current_file.data.loc[
            (self.current_file.data.et > time_interval.start_et) & (self.current_file.data.et < time_interval.end_et)
        ].copy()
        data["spot"] -= 1
        return data.to_dict(orient="records")

    def process_data_entry(
        self, data_entry: Dict, instrument: BaseInstrument, filter_obj: BaseFilter
    ) -> Optional[Dict]:
        et = data_entry["et"]
        self.kernel_manager.step_et(et)

        projection_vector = instrument.sub_instruments[data_entry["spot"]].transformed_boresight(instrument.frame, et)
        projection: ProjectionPoint = instrument.project_vector(et, projection_vector)

        if filter_obj.point_pass(projection.projection):
            data_entry.update(projection.to_data())
            return data_entry
        else:
            return None
