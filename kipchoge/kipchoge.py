from pathlib import Path
from typing import Union, List
import pandas as pd
import fitdecode
import os
import datetime
import logging
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

"""
`kipchoge` is my toy library for turning the .fit files 
from Garmin devices into a pandas dataframe.

requirements: pandas; fitdecode; tqdm

Usage:
```
kip = Kipchoge(fit_files = "path/to/your/file.fit")
or 
kip = Kipchoge(fit_files = ["path/to/your/file1.fit", "path/to/your/file2.fit"])

kip.analyze() # prints the statistics of the messages in the files
kip.as_dataframe(save_name = "data.csv") # parses the data and saves the data to a csv file
```

"""
_LOGGER = logging.getLogger(__name__)

# These are the messages that we are interested in
# the rest will be ignored
VALID_MESSAGE_NAMES = {
    'unknown_233',
    'gps_metadata',
    'record',
    'unknown_325'
    'unknown_297',
    'stress_level',
    'monitoring',
    'event'
}


class Kipchoge:
    def __init__(self, fit_files: Union[str, Path, List[str], List[Path]]):
        """
        :param fit_files: a single .fit file or a list of .fit files
        """
        if not isinstance(fit_files, list):
            fit_files = [fit_files]
        self.fit_files = [fitdecode.FitReader(fit_file) for fit_file in fit_files]

    def analyze(self, max_threads=10):
        """
        Analyzes the .fit files and prints the most frequent messages in the files
        """
        _LOGGER.info("Analyzing .fit files")
        with ThreadPoolExecutor(max_threads) as executor:
            counters = list(tqdm(executor.map(self.analyze_file, self.fit_files), total=len(self.fit_files)))
        counter_sum = pd.Series(sum(counters, Counter())).sort_values(ascending=False)
        _LOGGER.info("Printing the most frequent messages in the files")
        _LOGGER.info("------------------------------------------------")
        _LOGGER.info(counter_sum.to_string())

    @staticmethod
    def analyze_file(file: fitdecode.FitReader) -> Counter:
        """
        Analyzes a single .fit file and returns the most frequent messages in the file

        :param file: A fitdecode.FitReader object
        :return: A Counter object with counts of the message names in a file
        """
        valid_messages = (message for message in file if (message.frame_type == fitdecode.FIT_FRAME_DATA))
        message_names = Counter(message.name for message in valid_messages)
        return message_names

    def as_dataframe(self, save_name='kipchoge_data.csv', start_date: datetime.date = None,
                     end_date: datetime.date = None):
        """
        Parses the .fit files and saves the data to a csv file

        :param save_name: the name of the csv file to save the data to
        :param start_date: optional, the lower bound of the date range to include in the dataframe
        :param end_date: optional, the upper bound of the date range to include in the dataframe
        """
        dataframes = []
        # this should definitely be parallelized
        for file in tqdm(self.fit_files):
            raw_dataframe = self.process_file(file)
            dataframe = self.process_dataframe(raw_dataframe, start_date, end_date)
            dataframes.append(dataframe)
        df = pd.concat(dataframes)
        df.sort_values(by=['timestamp'], ascending=True)
        df.to_csv(save_name)

    @staticmethod
    def process_file(file: fitdecode.FitReader):
        """
        Processes a single .fit file and returns a pandas dataframe

        :param file: A fitdecode.FitReader object
        :return: A pandas dataframe
        """

        messages = list(message for message in file if
                        message.frame_type == fitdecode.FIT_FRAME_DATA and message.name in VALID_MESSAGE_NAMES)

        headers = [[field.name for field in message.fields if not field.name.startswith("unknown")] for message in
                   messages]
        units = [[field.units or "" for field in message.fields if not field.name.startswith("unknown")] for message in
                 messages]

        columns = [list(map(lambda x: x[0] + f" (" + x[1] + ")" if x[1] else x[0], zip(header, unit))) for header, unit
                   in zip(headers, units)]
        values = [[field.value for field in message.fields if not field.name.startswith("unknown")] for message in
                  messages]

        data_as_dict = [dict(zip(header, value)) for header, value in zip(columns, values)]
        return pd.DataFrame(data_as_dict)

    @staticmethod
    def process_dataframe(raw_dataframe: pd.DataFrame, start_date: datetime.date = None, end_date: datetime.date = None):
        """
        Processes a pandas dataframe and returns a cleaned dataframe

        :param raw_dataframe: a pandas dataframe
        :param start_date: optional, the lower bound of the date range to include in the dataframe
        :param end_date: optional, the upper bound of the date range to include in the dataframe
        :return: a cleaned pandas dataframe
        """
        if raw_dataframe.empty:
            return None

        # interpolate nans in timestamp column
        raw_dataframe["timestamp"] = raw_dataframe["timestamp"].interpolate(method='linear', limit_direction='both')
        # convert timestamp to datetime
        raw_dataframe["timestamp"] = pd.to_datetime(raw_dataframe["timestamp"])
        raw_dataframe = raw_dataframe.set_index('timestamp')
        raw_dataframe.index = pd.to_datetime(raw_dataframe.index)

        # filter by date
        if start_date is not None:
            raw_dataframe = raw_dataframe[raw_dataframe.index >= start_date]
        if end_date is not None:
            raw_dataframe = raw_dataframe[raw_dataframe.index <= end_date]

        if raw_dataframe.empty:
            return None

        # remove rows that have majority of nans
        raw_dataframe = raw_dataframe.dropna(thresh=int(0.1 * raw_dataframe.shape[1]), axis=0)
        # remove columns that have majority of nans
        raw_dataframe = raw_dataframe.dropna(thresh=int(0.1 * raw_dataframe.shape[0]), axis=1)
        # interpolate nans also backwards and forwards
        raw_dataframe = raw_dataframe.interpolate(method='linear', limit_direction='both')
        return raw_dataframe


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import glob

    kip = Kipchoge(glob.glob(os.path.join(
        "/Users/damian/Downloads/be58962e-ecf8-418c-91b2-19a7cb67480a_1/DI_CONNECT/DI-Connect-Uploaded-Files/",
        "*.fit")))
    kip.analyze()
    df = kip.as_dataframe(start_date="2023-11-22", end_date="2023-11-28")
