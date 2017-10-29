from datetime import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import callbacks
from keras import regularizers
from keras.engine import Model
from keras.layers import Input, Lambda, Concatenate, Dense, Reshape, Flatten
from keras.optimizers import SGD

TRAININGS_DATA_FILE = "./data/train_100.csv"
TEST_DATA_FILE = "./data/test_100.csv"
TEST_OUTPUT = "submission.csv"


def create_lookup(np_array):
    return tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            keys=np_array,
            values=np.arange(0, np_array.shape[0]),
        ),
        default_value=1
    )


if __name__ == "__main__":
    converters = {
        "onpromotion": (lambda p: int(p == "True")),
        "date": (lambda d: datetime.strptime(d, "%Y-%m-%d").weekday())
    }

    types = {
        'id': 'int64',
        'item_nbr': 'int64',
        'store_nbr': 'int64',
        'unit_sales': 'float32'
    }

    print("Loading Traing CSV ...")

    data = pd.read_csv(TRAININGS_DATA_FILE, dtype=types, converters=converters, keep_default_na=False, index_col=0)

    print("DONE")

    print("Preparing Look-Up-Tables ...")

    items = data['item_nbr'].drop_duplicates()
    item_lookup = create_lookup(items)
    item_num_classes = len(items)

    stores = data['store_nbr'].drop_duplicates()
    store_lookup = create_lookup(stores)
    store_num_classes = len(stores)

    K.get_session().run(tf.tables_initializer())

    print("DONE")

    print("Building Model ...")

    # Input Layer
    item_input = Input(shape=(1,), dtype='int64', name="item_id")
    day_input = Input(shape=(1,), dtype='uint8', name="day")
    store_input = Input(shape=(1,), dtype='int32', name="store_id")
    promotion_input = Input(shape=(1,), dtype='float32', name="promotion")

    # Encoding Layer
    item_normalize = Lambda(lambda t: item_lookup.lookup(tf.cast(t, 'int64')), name="item_normalize")(item_input)
    item_one_hot = Lambda(
        K.one_hot,
        arguments={'num_classes': item_num_classes},
        output_shape=(1, item_num_classes),
        name='item_encoding'
    )(item_normalize)

    day_one_hot = Lambda(
        K.one_hot,
        arguments={'num_classes': 7},
        output_shape=(1, 7),
        name='day_encoding'
    )(day_input)

    store_normalize = Lambda(lambda t: store_lookup.lookup(tf.cast(t, 'int64')), name="store_normalize")(store_input)
    store_one_hot = Lambda(
        K.one_hot,
        arguments={'num_classes': store_num_classes},
        output_shape=(1, store_num_classes),
        name='store_encoding'
    )(store_normalize)

    promotion_reshape = Reshape(target_shape=(1, 1), name="promotion_reshape")(promotion_input)

    # Concatenate all the input
    full_input = Concatenate()([item_one_hot, day_one_hot, store_one_hot, promotion_reshape])

    # Hidden layer
    hidden = Dense(64, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))(full_input)

    # Output layer
    output = Dense(1, activation="linear")(Flatten()(hidden))

    model = Model(inputs=[item_input, day_input, store_input, promotion_input], outputs=output)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="mse", metrics=["mae"])

    print("DONE")

    print(model.summary())

    print("Training model ...")

    model.fit(
        x=[data['item_nbr'], data['date'], data['store_nbr'], data['onpromotion']],
        y=data["unit_sales"],
        epochs=500,
        batch_size=100,
        callbacks=[
            callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        ],
    )

    print("DONE")

    print("Load Test CSV ...")

    test_data = pd.read_csv(
        TEST_DATA_FILE,
        dtype=types,
        converters=converters
    )

    print("DONE")

    print("Prediction ...")

    prediction = model.predict(
        x=[test_data['item_nbr'], test_data['date'], test_data['store_nbr'], test_data['onpromotion']],
    )

    submission = pd.concat([test_data['id'], pd.DataFrame(prediction, columns=['unit_sales'])], axis=1)
    submission.to_csv(TEST_OUTPUT, index=False)

    print("DONE")