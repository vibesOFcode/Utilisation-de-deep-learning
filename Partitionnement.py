import numpy as np
from tensorflow.keras.utils import to_categorical


def data_to_negative(data):
    """Invert pixel intensities to the [0, 1] range."""
    return 1.0 - np.asarray(data, dtype=np.float32)


def prepare_dataset(x_train, x_test, y_train, y_test, image_size, labels=None):
    """Reshape and normalize data for image classification."""
    if labels is None:
        labels = np.unique(y_train)

    x_train = data_to_negative(x_train).astype('float32') / 255.0
    x_test = data_to_negative(x_test).astype('float32') / 255.0

    x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1)
    x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1)

    y_train_categorical = to_categorical(y_train, num_classes=len(labels))
    y_test_categorical = to_categorical(y_test, num_classes=len(labels))

    return x_train, x_test, y_train_categorical, y_test_categorical, np.asarray(labels)


if __name__ == '__main__':
    x_train_example = np.random.rand(10, 64, 64).astype(np.float32)
    x_test_example = np.random.rand(3, 64, 64).astype(np.float32)
    y_train_example = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2, 1])
    y_test_example = np.array([0, 1, 2])

    x_train_processed, x_test_processed, y_train_cat, y_test_cat, labels = prepare_dataset(
        x_train_example,
        x_test_example,
        y_train_example,
        y_test_example,
        image_size=64,
        labels=np.array([0, 1, 2]),
    )

    print(x_train_processed.shape)
    print(y_train_cat.shape)
    print(labels)
