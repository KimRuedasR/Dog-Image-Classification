import os
import cv2
import imghdr
import tensorflow as tf
import matplotlib.pyplot as plt


# Directorio de datos y extensiones de imagen válidas
data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

# Función para eliminar imágenes inválidas
def remove_invalid_images(data_directory, valid_extensions):

    # Revisar el dataset y elimina imágenes que no estén en los formatos.
    for image_class in os.listdir(data_directory):
        class_dir = os.path.join(data_directory, image_class)
        if os.path.isdir(class_dir):
            for image in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image)
                try:
                    img = cv2.imread(image_path)
                    img_type = imghdr.what(image_path)
                    if img_type not in valid_extensions:
                        print(f'Imagen no en la lista de extensiones {image_path}')
                        os.remove(image_path)
                except Exception as e:
                    print(f'Problema con la imagen {image_path}')


# Función para cargar los datos
def load_data(directory):
    return tf.keras.utils.image_dataset_from_directory(directory)

# Inicializar las funciones
remove_invalid_images(data_dir, image_exts)
data = load_data(data_dir)

# Visualización de los datos
def display_batch(batch):
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])
    plt.show()

# Obtener un batch de datos
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
display_batch(batch)
