# 7. Model Training
import tensorflow as tf
from model_building import build_model
import os

# Configuración de variables de entorno para CUDA y XLA para asegurar el uso de aceleración por GPU
os.environ['CUDA_HOME'] = '/opt/cuda'
os.environ['LD_LIBRARY_PATH'] = '/opt/cuda/lib64:/opt/cuda/extras/CUPTI/lib64'
os.environ['PATH'] = '/opt/cuda/bin:' + os.environ['PATH']
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/cuda'

# Función para cargar los datos
def load_data(data_dir, img_height=256, img_width=256, batch_size=32, subset=None):
    if subset == 'training':
        return tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
    elif subset == 'validation':
        return tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
    elif subset == 'test':
        return tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size)
    else:
        raise ValueError("Set invalido, elegir: 'training', 'validation', o 'test'")

# Función para entrenar el modelo
def train_model(train_ds, val_ds, epochs=20):
    # Construir el modelo
    model = build_model()  # Asegúrate de haber importado build_model de model_building.py
    # Callback para TensorBoard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
    # Entrenar el modelo
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[tensorboard_callback])

    model.save('models/imageclassifier.h5')

    return model, history

# Cargar los datos de entrenamiento y entrenar el modelo
if __name__ == "__main__":
    train_data = load_data('data/dogs', subset='training')
    val_data = load_data('data/dogs', subset='validation')
    model, history = train_model(train_data, val_data)
