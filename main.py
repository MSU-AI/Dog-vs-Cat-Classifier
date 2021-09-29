import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array

def plot_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def train_model():
    train_gen = ImageDataGenerator(
        rescale=1. / 255,
    )

    train_iterator = train_gen.flow_from_directory('train',
                                                   target_size=(150, 150),
                                                   batch_size=20,
                                                   class_mode='binary')

    validation_gen = ImageDataGenerator(rescale=1. / 255.0)
    validation_iterator = validation_gen.flow_from_directory('validate',
                                                             target_size=(150, 150),
                                                             batch_size=10,
                                                             class_mode='binary')

    model = keras.models.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        
        keras.layers.MaxPool2D((2, 2)),
        
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        
        keras.layers.MaxPool2D((2, 2)),
        
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        
        keras.layers.MaxPool2D((2, 2)),
        
        keras.layers.Flatten(),
        
        keras.layers.Dense(units=512, activation="relu"),
        
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])

    history = model.fit(train_iterator,
                        validation_data=validation_iterator,
                        steps_per_epoch=1000,
                        epochs=50,
                        validation_steps=500)
    
    model.save('model.h5')

    return history


def load_and_predict(filename):
    model = keras.models.load_model('model.h5')
    
    image = load_img(f'test/{filename}', target_size=(150,150))
    image = img_to_array(image)
    image /= 255
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    prediction = model.predict(image)
    
    return prediction
    
def main():
    while True:
        filename = input('Input the name of the file you want to classify ("q" to quit): ')
        if filename == "q":
            break
        
        try:
            prediction = load_and_predict(filename)[0][0]
        except:
            print("No such file found")
            continue
        
        if prediction < 0.3:
            print("It's a cat!")
        elif prediction > 0.7:
            print("It's a dog!")
        else:
            print("I'm quite not sure.")
        print("Prediction Value (where 0 means cat and 1 means dog):", prediction)
    
main()