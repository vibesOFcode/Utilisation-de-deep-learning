from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense

from geometric_model import build_geometric_model
from Spectral_model import build_spectral_model


def build_fusion_model(input_shape=(128, 128, 1), spectral_shape=(128, 128, 1), num_classes=10):
    """Fuse the spatial and spectral CNN branches into one model."""
    geometric_branch = build_geometric_model(input_shape=input_shape, num_classes=num_classes)
    spectral_branch = build_spectral_model(input_shape=spectral_shape, num_classes=num_classes)

    merged = Concatenate(name='fusion')([geometric_branch.output, spectral_branch.output])
    x = Dense(256, activation='relu')(merged)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax', name='final_output')(x)

    model = Model(
        inputs=[geometric_branch.input, spectral_branch.input],
        outputs=outputs,
        name='fusion_model',
    )
    return model


fusion_model = build_fusion_model()


if __name__ == '__main__':
    fusion_model.summary()