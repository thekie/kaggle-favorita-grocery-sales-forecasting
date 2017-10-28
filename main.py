import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

TRAININGS_DATA_FILE = "./processed_data/data_100.csv"

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

    model = Sequential([
        Dense(128, input_dim=normalized_data.shape[1], activation="sigmoid"),
        Dense(1, activation="linear")
    ])

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="mse", metrics=["accuracy"],)

    model.fit(x=normalized_data.values, y=sales.values, epochs=1000, batch_size=100)

    prediction = model.predict(normalized_data.values)

    prediction_data = pd.DataFrame(
        data=prediction,
        columns=["unit_sales"]
    )

    submission_data = pd.concat([data["id"], prediction_data], axis=1)
    submission_data.to_csv("submission.csv", index=False)