from math import ceil

import pandas as pd
from keras import callbacks
from keras.layers import Dense, regularizers
from keras.models import Sequential
from keras.optimizers import SGD

TRAININGS_DATA_FILE = "./processed_data/data_10000.h5"
BATCH_SIZE = 1000
STORE = pd.HDFStore(TRAININGS_DATA_FILE)
M = STORE.get_storer("ids").shape[0]


def chunker():
    i = 0
    while True:
        items = STORE.select("items", start=i * BATCH_SIZE, stop=(i + 1) * BATCH_SIZE - 1)
        days = STORE.select("days", start=i * BATCH_SIZE, stop=(i + 1) * BATCH_SIZE - 1)
        stores = STORE.select("stores", start=i * BATCH_SIZE, stop=(i + 1) * BATCH_SIZE - 1)
        sales = STORE.select("sales", start=i * BATCH_SIZE, stop=(i + 1) * BATCH_SIZE - 1)
        yield pd.concat([items, days, stores], axis=1), sales
        i = (i+1) % ceil(M/BATCH_SIZE)

if __name__ == "__main__":
    tb_cb = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                  write_graph=True, write_images=True)

    input_dim = STORE.get_storer("items").shape[1] + STORE.get_storer("days").shape[1] + STORE.get_storer("stores").shape[1]

    model = Sequential([
        Dense(64, input_dim=input_dim, activation="sigmoid", kernel_regularizer=regularizers.l2(0.001)),
        Dense(1, activation="linear")
    ])

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="mse", metrics=["mae"])

    model.fit_generator(chunker(), epochs=500, callbacks=[tb_cb], steps_per_epoch=ceil(M/BATCH_SIZE))

    #prediction = model.predict(one_chunk)

    #prediction_data = pd.DataFrame(
    #    data=prediction,
    #    columns=["unit_sales"]
    #)

    #submission_data = pd.concat([data["id"], prediction_data], axis=1)
    #submission_data.to_csv("submission.csv", index=False)