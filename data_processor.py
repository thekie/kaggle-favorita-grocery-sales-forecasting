import pandas as pd
from datetime import datetime


INPUT_FILE = "./data/train_10000.csv"
OUTPUT_FILE = "./processed_data/data_10000.h5"

if __name__ == "__main__":

    print("Data pre-preprocessing started ...")
    print("Loading CSV ... ")

    data = pd.read_csv(
        INPUT_FILE,
        converters={
            "onpromotion": (lambda p: int(p == "True")),
            "date": (lambda d: datetime.strptime(d, "%Y-%M-%d").weekday())
        },
        keep_default_na=False,
        index_col=0,
        engine="c",
        low_memory=False
    )

    print("DONE")

    print("Writing to HDF Store ...")

    data.to_hdf(OUTPUT_FILE, "data", mode="w", format="table")
    del data

    print("DONE")

    print("Loading HDF Store ... ")
    hdf = pd.HDFStore(OUTPUT_FILE)
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
    normalized_data.to_hdf(OUTPUT_FILE, "data", mode="w", format="table")

    print("DONE")

    print(normalized_data.head())
