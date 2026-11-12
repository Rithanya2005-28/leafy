"""Re-evaluate the saved model without retraining (reproducibility check)."""

import json
import os

import numpy as np
import tensorflow as tf

import train_model as tm

model = tf.keras.models.load_model(tm.MODEL_PATH)
train_gen, validation_gen, test_gen = tm.make_generators()

print('model file mtime', os.path.getmtime(tm.MODEL_PATH), 'size', os.path.getsize(tm.MODEL_PATH))
print('class order first/last', list(test_gen.class_indices)[:2], list(test_gen.class_indices)[-2:])

test_gen.reset()
probs = model.predict(test_gen, verbose=0)
y_pred = np.argmax(probs, axis=1)
y_true = test_gen.classes
print('test acc from saved model:', float(np.mean(y_pred == y_true)))

validation_gen.reset()
vprobs = model.predict(validation_gen, verbose=0)
vpred = np.argmax(vprobs, axis=1)
print('validation acc from saved model:', float(np.mean(vpred == validation_gen.classes)))

with open(tm.METRICS_PATH, encoding='utf-8') as handle:
    stored = json.load(handle)
print('stored metrics:', stored['test_accuracy'], stored['test_samples'])
