import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Initialize an object that automatically formats the image data
ImageGenerator1 = ImageDataGenerator(rescale=1. / 255)
ImageGenerator2 = ImageDataGenerator(rescale=1. / 255)

def train_model():
    # Store all the images in train folder to train_images object,
    # using ImageGenerator1 as constructor.
    # Specify the size you want to resize all images to
    train_images = ImageGenerator1.flow_from_directory('train',
                                                   target_size=(150, 150),
                                                   class_mode='binary')
    
    # Store all the images in validation folder to validation_images object,
    # using ImageGenerator2 as constructor.
    # Specify the size you want to resize all images to
    validation_images = ImageGenerator2.flow_from_directory('validate',
                                                             target_size=(150, 150),
                                                             class_mode='binary')

    # Start building your Neural Network
    model = keras.models.Sequential([
        # Input layer and first convolutional layer of 16 filters
        keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        
        # Apply pooling to your image
        keras.layers.MaxPool2D((2, 2)),
        
        # Second convolutional layer of 32 filters
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        
        # Apply pooling to your image
        keras.layers.MaxPool2D((2, 2)),
        
        # Third convolutional later of 64 filters
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        
        # Apply pooling to your image
        keras.layers.MaxPool2D((2, 2)),
        
        # Flatten your image
        keras.layers.Flatten(),
        
        # Hidden layer of 512 neurons
        keras.layers.Dense(units=512, activation="relu"),
        
        # Output layer of 1 neuron (goes from 0 to 1 where 0 is cat and 1 is dog)
        keras.layers.Dense(1, activation="sigmoid")
    ])
    
    # Compile your neural network
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=['accuracy'])
    
    # Fit your model with the training images, and validate with the validation images
    model.fit(train_images,
            validation_data=validation_images,
            steps_per_epoch=1000,
            epochs=50,
            validation_steps=500)
    
    model.save('model.h5')

    return model


def load_and_predict(filename):
    try:
        model = keras.models.load_model('model.h5')
    except:
        model = train_model()
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