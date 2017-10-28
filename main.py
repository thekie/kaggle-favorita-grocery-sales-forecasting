import pandas as pd
from keras import callbacks
from keras.layers import Dense, regularizers
from keras.models import Sequential
from keras.optimizers import SGD

TRAININGS_DATA_FILE = "./processed_data/data_10000.csv"

if __name__ == "__main__":
    data = pd.read_csv(
        TRAININGS_DATA_FILE,
        usecols=["id", "date", "store_nbr", "item_nbr", "onpromotion", "unit_sales"]
    )

    print(data.head())
    day = pd.get_dummies(data["date"])
    item = pd.get_dummies(data["item_nbr"])
    store = pd.get_dummies(data["store_nbr"])
    promotion = data["onpromotion"]
    sales = data["unit_sales"]

    normalized_data = pd.concat([day, store, item, promotion], axis=1)
    print(normalized_data.head())

    tb_cb = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                write_graph=True, write_images=True)

    model = Sequential([
        Dense(64, input_dim=normalized_data.shape[1], activation="sigmoid", kernel_regularizer=regularizers.l2(0.001)),
        Dense(1, activation="linear")
    ])

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="mse", metrics=["mae"],)

    model.fit(x=normalized_data.values, y=sales.values, epochs=500, batch_size=100, callbacks=[tb_cb])

    prediction = model.predict(normalized_data.values)

    prediction_data = pd.DataFrame(
        data=prediction,
        columns=["unit_sales"]
    )

    submission_data = pd.concat([data["id"], prediction_data], axis=1)
    submission_data.to_csv("submission.csv", index=False)