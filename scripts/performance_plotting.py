import matplotlib.pyplot as plt

# Graficar el rendimiento del modelo
def plot_performance(history):
    # Graficar la pérdida
    fig = plt.figure()
    plt.plot(history.history['loss'], color='teal', label='loss')
    plt.plot(history.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Pérdida', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    # Graficar la precisión
    fig = plt.figure()
    plt.plot(history.history['accuracy'], color='teal', label='accuracy')
    plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Precisión', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
# Inicializar funciones y pasar el objeto 'history'
if __name__ == "__main__":
    history = ...
    plot_performance(history)
