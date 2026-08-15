import numpy as np
import cv2
from tensorflow.keras.utils import Sequence, to_categorical

N_CHANNELS_SPECTRAL = 1


def preprocess_and_get_spectrum(image_path, image_size):
    """Compute a normalized frequency spectrum from an image."""
    try:
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return None

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        img_resized = cv2.resize(img_gray, (image_size[1], image_size[0]))
        f_transform = np.fft.fft2(img_resized)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_transform_shifted)
        magnitude_spectrum_log = np.log1p(magnitude_spectrum)

        min_value = np.min(magnitude_spectrum_log)
        max_value = np.max(magnitude_spectrum_log)

        if max_value - min_value > 1e-6:
            spectrum_normalized = (magnitude_spectrum_log - min_value) / (max_value - min_value)
        else:
            spectrum_normalized = np.zeros_like(magnitude_spectrum_log)

        return np.expand_dims(spectrum_normalized, axis=-1).astype("float32")
    except Exception as exc:
        print(f"Error processing {image_path}: {exc}")
        return None


class SpectralDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, image_size, num_classes, shuffle=True):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.batch_size = batch_size
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[k] for k in batch_indexes]
        batch_labels = [self.labels[k] for k in batch_indexes]

        batch_size = len(batch_indexes)
        X_batch = np.empty((batch_size, self.image_size[0], self.image_size[1], N_CHANNELS_SPECTRAL), dtype="float32")
        y_batch = np.zeros((batch_size, self.num_classes), dtype="float32")

        for i, path in enumerate(batch_image_paths):
            spectrum = preprocess_and_get_spectrum(path, self.image_size)
            if spectrum is not None:
                X_batch[i] = spectrum
            else:
                X_batch[i] = np.zeros((self.image_size[0], self.image_size[1], N_CHANNELS_SPECTRAL), dtype="float32")

            y_batch[i] = to_categorical(batch_labels[i], num_classes=self.num_classes)

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
