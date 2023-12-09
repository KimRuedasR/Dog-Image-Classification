# 2. Data scaling
import tensorflow as tf

# Escalar los datos
def scale_data(data):
    # Escala los datos para normalizar los valores
    return data.map(lambda x, y: (x / 255, y))

# Cargar los datos
data_dir = 'data'
data = tf.keras.utils.image_dataset_from_directory(data_dir)

# Aplicar la función de escalado
scaled_data = scale_data(data)

# Verificar un ejemplo del conjunto de datos escalado*/<,
scaled_data_iterator = scaled_data.as_numpy_iterator()
batch = scaled_data_iterator.next()

# Imprime las dimensiones del batch para verificar
print(batch[0].shape, batch[1].shape)  
