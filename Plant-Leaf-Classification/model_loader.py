import os

def load_model(model_path='model/leaf_classifier.keras'):
    """Load a Keras model from the given path.
    If the model file does not exist, raise FileNotFoundError with a clear
    message instructing the user to train the model first.
    TensorFlow is imported lazily so the web server can still start without it.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'.\n"
            "Please run 'python train_model.py' to train and save the model."
        )
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)

