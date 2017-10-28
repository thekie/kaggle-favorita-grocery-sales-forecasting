import tensorflow as tf
import numpy as np

TRAININGS_DATA_FILE = "./processed_data/data_10000.csv"

if __name__ == "__main__":
    trainings_data = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=TRAININGS_DATA_FILE,
        target_column=3,
        target_dtype=np.float32,
        features_dtype=np.float32
    )
    trainings_np_data = np.array(trainings_data.data)[:, 1:]

    feature_columns = [tf.feature_column.numeric_column("x", shape=(4,))]
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": trainings_np_data},
        y=np.array(trainings_data.target),
        num_epochs=10000,
        shuffle=True
    )

    estimator.train(train_input_fn)
    print(estimator.evaluate(train_input_fn))

    new_sample = np.array([[1, 25, 7.0, 0]], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_sample},
        num_epochs=1,
        shuffle=False
    )

    prediction = estimator.predict(predict_input_fn)
    print([p for p in prediction])