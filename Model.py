import tensorflow
from keras import Input, Model
from keras.src.layers import Reshape, SimpleRNN, LSTM, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D,BatchNormalization,Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras import regularizers

class Autoencoder(tensorflow.keras.Model):
    # Set up image data generators for train and test datasets
    def anmodel(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(28, 28),
        batch_size=32,
        class_mode='input' )

        test_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(28, 28),
        batch_size=32,
        class_mode='input' )
        input_img = Input(shape=(28, 28, 3))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='softmax', padding='same')(x)
        # Create autoencoder model
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
        return autoencoder, train_generator,test_generator

class DeepANN:
    def rnn(self,input_shape=(128,128,3),num_class=3):
        model=Sequential()
        # size : 128,128,3  ==>  128*128,3
        model.add(Reshape((input_shape[0]*input_shape[1], input_shape[2]), input_shape=input_shape))
        model.add(SimpleRNN(128))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(num_class,activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        return model

    def LSTM_model(self,input_shape=(128,128,3),num_class=3):
        model=Sequential()
        # size : 128,128,3  ==>  128*128,3
        model.add(Reshape((input_shape[0]*input_shape[1], input_shape[2]), input_shape=input_shape))
        model.add(LSTM(128))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(num_class,activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        return model

    def cnn_vgg(self):
        model = Sequential()
        model.add(VGG16(include_top=False,weights='imagenet', input_shape=(128,128, 3)))
        # fully connected layer
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # output layer
        model.add(Dense(3, activation="softmax"))
        # compilation
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model

    def cnn_model(self, input_shape=(128,128, 3), optimizer='sgd'):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128,128, 3)))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(3, activation="softmax"))  # Use softmax for multiclass classification

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        return model

    def cnn_add_regularize(self):
        model = Sequential()
        # convolutional layers
        model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128,128, 3)))
        # in input  shape 3 represent  coloured images
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        # fully connected layer
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # output layer
        model.add(Dense(3, activation='softmax'))
        # compilation
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model

    def simple_ANN(self, input_shape=(128,128, 3), optimizer='sgd'):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(3, activation="softmax"))  # Use softmax for multiclass classification

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model



def train_model(model_instance, tr_gen, va_gen, epochs=10):
    history = model_instance.fit(tr_gen, validation_data=va_gen, epochs=epochs)
    return history

def compare_model(models,tr_gen,va_gen,tt_gen,epochs=5):
    histories=[]

    for model in models:
        history=train_model(model,tr_gen,va_gen,epochs=epochs)
        mo_loss, mo_acc = model.evaluate(tt_gen)
        print("the ann Architecture")
        print(model.summary())
        print(f"test accuracy: {mo_acc}")
        histories.append(history)
        #plotting
    fig, axes = plt.subplots(nrows=2, figsize=(10, 12))
    for i, history in enumerate(histories):
        axes[0].plot(history.history['accuracy'], label=i)
        axes[1].plot(history.history['loss'], label=i)
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[1].set_title('Model Loss Comparison')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    plt.tight_layout()
    plt.show(block=True)

def compare_model1(models,mo,tr_gen, tt_gen, va_gen,epochs=5):
    histories=[]
    #mo = ['ann_adam', 'ann_sgd', 'ann_rmsprop', 'cnn_adam', 'cnn_sgd', 'cnn_rmsprop']
    for model in models:
        history=train_model(model,tr_gen,va_gen,epochs=epochs)
        print("the ann Architecture")
        print(model.summary())
        print('accuracy : ',model.evaluate)
        mo_loss, mo_acc = model.evaluate(tt_gen)
        print(f"test accuracy: {mo_acc}")
        histories.append(history)
        #plotting
    fig, axes = plt.subplots(nrows=2, figsize=(10, 12))
    for i, history in enumerate(histories):
        axes[0].plot(history.history['accuracy'], label=mo[i])
        axes[1].plot(history.history['loss'], label=mo[i])
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].set_title('Model Loss Comparison')
    axes[1].set_xlabel('Epochs')
    #axes[1].set_ylabel('Loss')
    axes[1].legend()
    plt.savefig('static/images/compare.jpg')


