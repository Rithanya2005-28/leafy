"""Train a leaf-species classifier on the prepared dataset.

Run `python prepare_dataset.py` first. This script:
  1. loads dataset_split/train and dataset_split/validation
  2. trains a MobileNetV2 transfer-learning model
  3. evaluates on dataset_split/test
  4. saves model/leaf_classifier.keras, model/class_names.json,
     model/metrics.json, model/confusion_matrix.png
"""

import json
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(BASE_DIR, 'dataset_split')
TRAIN_DIR = os.path.join(SPLIT_DIR, 'train')
VALIDATION_DIR = os.path.join(SPLIT_DIR, 'validation')
TEST_DIR = os.path.join(SPLIT_DIR, 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'leaf_classifier.keras')
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.json')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
CONFUSION_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
HEAD_EPOCHS = 12
FINE_TUNE_EPOCHS = 4
SEED = 42


def preprocess_for_model(image):
    """Shared inference/training preprocessing: resize is done by Keras,
    then Gaussian blur (noise reduction) and MobileNetV2 scaling to [-1, 1]."""
    image = np.clip(image, 0, 255).astype('uint8')
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image.astype('float32') / 127.5 - 1.0


def make_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_for_model,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode='nearest',
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_for_model)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=SEED,
    )
    validation_gen = eval_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )
    return train_gen, validation_gen, test_gen


def build_model(num_classes):
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    )
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model, base


def evaluate_model(model, test_gen, class_names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    test_gen.reset()
    probabilities = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_gen.classes

    top1 = float(np.mean(y_pred == y_true))
    top3_matches = np.any(
        np.argsort(probabilities, axis=1)[:, -3:] == y_true[:, None], axis=1
    )
    top3 = float(np.mean(top3_matches))

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        zero_division=0, output_dict=True,
    )

    np.savetxt(
        os.path.join(MODEL_DIR, 'confusion_matrix.csv'),
        matrix, fmt='%d', delimiter=',',
    )

    fig, ax = plt.subplots(figsize=(20, 20))
    image = ax.imshow(matrix, interpolation='nearest', cmap='Greens')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=f'Confusion matrix (test accuracy {top1:.1%})',
    )
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=150)
    plt.close(fig)

    metrics = {
        'test_accuracy': top1,
        'test_top3_accuracy': top3,
        'test_samples': int(len(y_true)),
        'num_classes': len(class_names),
        'per_class': {
            name: {
                'precision': report[name]['precision'],
                'recall': report[name]['recall'],
                'f1': report[name]['f1-score'],
                'support': report[name]['support'],
            }
            for name in class_names
        },
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
    }
    with open(METRICS_PATH, 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    print(f'\nTest accuracy: {top1:.2%}')
    print(f'Top-3 accuracy: {top3:.2%}')
    print(f'Macro F1: {metrics["macro_f1"]:.3f}')
    print(f'Confusion matrix saved to {CONFUSION_PATH}')
    print(f'Metrics saved to {METRICS_PATH}')
    return metrics


def main():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            'dataset_split/train not found. Run "python prepare_dataset.py" first.'
        )
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_gen, validation_gen, test_gen = make_generators()
    class_names = sorted(train_gen.class_indices, key=train_gen.class_indices.get)

    # Keep the class order used by the model, so app predictions line up.
    with open(CLASS_NAMES_PATH, 'w', encoding='utf-8') as handle:
        json.dump(class_names, handle, indent=2)

    print(f'Classes: {len(class_names)}')
    print(f'Train: {train_gen.samples} | Validation: {validation_gen.samples} | Test: {test_gen.samples}')

    model, base = build_model(len(class_names))
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    print('\nStage 1: training the classification head (frozen MobileNetV2)')
    model.fit(
        train_gen,
        epochs=HEAD_EPOCHS,
        validation_data=validation_gen,
        callbacks=callbacks,
        verbose=1,
    )

    print('\nStage 2: fine-tuning the top MobileNetV2 layers')
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(
        train_gen,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=validation_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # Load the best checkpoint before evaluating.
    best_model = tf.keras.models.load_model(MODEL_PATH)
    evaluate_model(best_model, test_gen, class_names)
    print(f'\nModel saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()
