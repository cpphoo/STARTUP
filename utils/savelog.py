import datetime
import time
import os

import pandas as pd

__all__ = ['savelog']

class savelog:
    ''' Saves training log to csv'''
    INCREMENTAL_UPDATE_TIME = 0

    def __init__(self, directory, name):
        self.file_path = os.path.join(directory, "{}_{:%Y-%m-%d_%H:%M:%S}.csv".format(name, datetime.datetime.now()))
        self.data = {}
        self.last_update_time = time.time() - self.INCREMENTAL_UPDATE_TIME

    def record(self, step, value_dict):
        self.data[step] = value_dict
        if time.time() - self.last_update_time >= self.INCREMENTAL_UPDATE_TIME:
            self.last_update_time = time.time()
            self.save()

    def save(self):
        df = pd.DataFrame.from_dict(self.data, orient='index').to_csv(self.file_path)