import os

import pandas as pd
from datetime import datetime


INPUT_FILE = "./data/train_10000.csv"
OUTPUT_FILE = "./processed_data/data_10000.h5"
CHUNKSIZE = 50000

if __name__ == "__main__":

    print("Data pre-preprocessing started ...")

    os.remove(OUTPUT_FILE)

    print("Loading CSV into HDF Store ... ")

    hdf = pd.HDFStore(OUTPUT_FILE)

    converters = {
        "onpromotion": (lambda p: int(p == "True")),
        "date": (lambda d: datetime.strptime(d, "%Y-%M-%d").weekday())
    }

    types = {
        'id': 'int64',
        'item_nbr': 'int32',
        'store_nbr': 'int16',
        'unit_sales': 'float32',
        'onpromotion': bool,
    }

    i = 0
    for chunk in pd.read_csv(INPUT_FILE, dtype=types, converters=converters, keep_default_na=False, index_col=0, chunksize=CHUNKSIZE):
        hdf.append("data", chunk, index=False)
        i += 1
        print(i*CHUNKSIZE)
    hdf.create_table_index("data")

    print("DONE")

    print("Normalizing Data ... ")
    id = hdf.select("data", where="columns = ['id']")
    day = pd.get_dummies(hdf.select("data", where="columns = ['date']"), columns=["date"])
    item = pd.get_dummies(hdf.select("data", where="columns = ['item_nbr']"), columns=["item_nbr"])
    store = pd.get_dummies(hdf.select("data", where="columns = ['store_nbr']"), columns=["store_nbr"])
    promotion = hdf.select("data", where="columns = ['onpromotion']")
    sales = hdf.select("data", where="columns = ['unit_sales']")
    hdf.close()

    del hdf

    print("DONE")

    print("Writing normalized data ... ")

    normalized_data = pd.concat([id, day, store, item, promotion, sales], axis=1)
    normalized_data.to_hdf(OUTPUT_FILE, "normalized_data", mode="w", format="table")

    print("DONE")

    print(normalized_data.head())
