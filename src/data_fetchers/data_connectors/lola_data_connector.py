import os
import io
import pvl
import zipfile
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
from src.data_fetchers.config import LOLA_BASE_URL, LOLA_LBL_FILE_DUMP, LOLA_REMOTE_CACHE_FILE, LOLA_DATASET_ROOT
from src.SPICE.config import LRO_INT_ID
from src.SPICE.utils import NoDatetimeDecoder
from src.global_config import SUPRESS_TQDM
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




class LOLADataConnector(BaseDataConnector):

    name = "LOLARDR"

    timeseries = {
        "timeField": "et",
        "metaField": "meta",
        "granularity": "seconds",
    }

    indices = []

    @property
    def dataset_structure(self):
        """Returns all filer mapped to inclusive time intervals they should cover"""
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
        dataset_slice = []
        for time_interval, file_dict in self.dataset_structure:
            if time_interval[1] >= time_intervals.start_et and time_interval[0] <= time_intervals.end_et:
                dataset_slice.append(file_dict)

        lbl_tasks: List[Tuple[Dict[str, str], Future]] = []
        for file_dict in dataset_slice:
            lbl_tasks.append((file_dict, self.fetch_url_async(compose_lola_url(file_dict["lbl"]))))


        virtual_files: List[VirtualFile] = []
        for file_dict, fut in tqdm(lbl_tasks, desc="Downloading metadata", disable=SUPRESS_TQDM):
            resp = fut.result()
            meta = pvl.loads(resp.text, decoder=NoDatetimeDecoder())
            interval = TimeInterval(meta["START_TIME"], meta["STOP_TIME"])
            virtual_files.append(VirtualFile(compose_lola_url(file_dict["dat"]), interval, meta))

        virtual_files.sort(key=lambda x: x.interval)
        remote_interval_list = IntervalList([file.interval for file in virtual_files])
        mask = remote_interval_list.intersection_mask(time_intervals)
        virtual_files = [vf for keep, vf in zip(mask, virtual_files) if keep]
        return virtual_files

    def _parse_current_file(self):
        # In case the current file is not downloaded yet, wait until it is
        self.current_file.wait_to_be_downloaded()
        # Assume self.current_file.file is fully loaded
        with zipfile.ZipFile(self.current_file.file, "r") as zip_file:
            # In DIVINER dataset, there is just a single data file within the ZIP archive
            with zip_file.open(zip_file.filelist[0]) as f:
                df = pd.read_csv(
                    io.StringIO(f.read().decode("utf-8")),
                    skiprows=3,
                    delimiter=",",
                    usecols=list(dtypes.keys()),
                    skipinitialspace=True,
                )

                def sclk_float_to_scs2e_str(arr: np.ndarray) -> List[str]:
                    sec = np.floor(arr).astype(int)
                    ticks = np.round((arr - sec) * 65536).astype(int)
                    return [f"{s}:{t}" for s, t in zip(sec, ticks)]

                sclk_strings = sclk_float_to_scs2e_str(df["sclk"].values)
                df["et"] = [spice.scs2e(LRO_INT_ID, sclk) for sclk in sclk_strings]
                df.sort_values("et", inplace=True)
                self.current_file.data = df

    def _get_interval_data_from_current_file(self, time_interval: TimeInterval):
        # We assume the file is parsed, as is implemented in _load_next_file method logic
        data = self.current_file.data.loc[
            (self.current_file.data.et > time_interval.start_et) & (self.current_file.data.et < time_interval.end_et)
        ].copy()
        data["c"] = -1
        data["det"] = -1
        data[["cx_dsk", "cy_dsk", "cz_dsk"]] = self.kernel_manager.dsk.latlon_to_cartesian(data["clat"], data["clon"])
        return data.to_dict(orient="records")

    def process_data_entry(
        self, data_entry: Dict, instrument: BaseInstrument, filter_obj: BaseFilter
    ) -> Optional[Dict]:
        et = data_entry["et"]
        self.kernel_manager.step_et(et)
        projection_vector = (
            instrument.sub_instruments[data_entry["c"]]
            .pixels[data_entry["det"]]
            .transformed_boresight(instrument.frame, et)
        )

        projection: ProjectionPoint = instrument.project_vector(et, projection_vector)
        if filter_obj.point_pass(projection.projection):
            data_entry.update(projection.to_data())
            return data_entry
        else:
            return None
