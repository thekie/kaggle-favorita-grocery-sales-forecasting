import os

import pandas as pd
from datetime import datetime


INPUT_FILE = "./data/train_100.csv"
OUTPUT_FILE = "./processed_data/data_100.h5"
CHUNKSIZE = 50000

if __name__ == "__main__":

    print("Data pre-preprocessing started ...")

    os.remove(OUTPUT_FILE)

    print("Loading CSV into HDF Store ... ")

    hdf = pd.HDFStore(OUTPUT_FILE)

    print("Loading Column 'id' ... ")
    df = pd.read_csv(INPUT_FILE, usecols=["id"], dtype='int64')
    print("DONE")

    print("Saving Column 'id' ... ")
    df.to_hdf(OUTPUT_FILE, "ids")
    del df
    print("DONE")

    print("Loading Column 'date' ... ")
    df = pd.read_csv(INPUT_FILE, usecols=["date"], converters={
        "date": (lambda d: datetime.strptime(d, "%Y-%M-%d").weekday())
    })
    df = pd.get_dummies(df["date"])
    print("DONE")

    print("Saving Column 'date' ... ")
    df.to_hdf(OUTPUT_FILE, "days")
    del df
    print("DONE")

    print("Loading Column 'item_nbr' ... ")
    df = pd.read_csv(INPUT_FILE, usecols=["item_nbr"], dtype='int32')
    df = pd.get_dummies(df["item_nbr"])
    print("DONE")

    print("Saving Column 'item_nbr' ... ")
    df.to_hdf(OUTPUT_FILE, "items")
    del df
    print("DONE")

    print("Loading Column 'store_nbr' ... ")
    df = pd.read_csv(INPUT_FILE, usecols=["store_nbr"], dtype='int16')
    df = pd.get_dummies(df["store_nbr"])
    print("DONE")

    print("Saving Column 'store_nbr' ... ")
    df.to_hdf(OUTPUT_FILE, "stores")
    del df
    print("DONE")

    print("Loading Column 'onpromotion' ... ")
    df = pd.read_csv(INPUT_FILE, usecols=["onpromotion"], converters={
        "onpromotion": (lambda p: int(p == "True")),
    })
    print("DONE")

    print("Saving Column 'onpromotion' ... ")
    df.to_hdf(OUTPUT_FILE, "sales")
    del df
    print("DONE")

    print("Loading Column 'unit_sales' ... ")
    df = pd.read_csv(INPUT_FILE, usecols=["unit_sales"], dtype='float32')
    print("DONE")

    print("Saving Column 'unit_sales' ... ")
    df.to_hdf(OUTPUT_FILE, "sales")
    del df
    print("DONE")

    print(hdf)
    hdf.close()
