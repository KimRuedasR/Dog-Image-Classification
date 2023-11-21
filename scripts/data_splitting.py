import tensorflow as tf

# Cargar los datos escalados
data_dir = 'data'
data = tf.keras.utils.image_dataset_from_directory(data_dir)

# Dividir los datos
def split_data(data, train_size=0.7, val_size=0.2, test_size=0.1):
    # Divide los datos en conjuntos de entrenamiento, validación y prueba.
    total_size = len(data)
    train_data = data.take(int(total_size * train_size))
    val_data = data.skip(int(total_size * train_size)).take(int(total_size * val_size))
    test_data = data.skip(int(total_size * (train_size + val_size))).take(int(total_size * test_size))
    return train_data, val_data, test_data

# Aplicar la función de división
train, val, test = split_data(data)

# Verificar las longitudes de cada conjunto
print(f"Entrenamiento: {len(train)}, Validación: {len(val)}, Prueba: {len(test)}")
