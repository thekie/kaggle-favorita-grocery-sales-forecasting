import pandas as pd
from datetime import datetime

INPUT_FILE = "./data/train_100.csv"
OUTPUT_FILE = "./processed_data/data_100.csv"

if __name__ == "__main__":
    data = pd.read_csv(
        INPUT_FILE,
        converters={
            "onpromotion": (lambda p: int(p == "True")),
            "date": (lambda d: datetime.strptime(d, "%Y-%M-%d").weekday())
        },
        keep_default_na=False,
        index_col=0
    )

    data.to_csv(OUTPUT_FILE)
