from keras.src.saving.saving_api import load_model

import preprocessData as pp
import matplotlib.pyplot as plt
import Model as models

obj= pp.preprocess_data()

#directory path
dir_path="train"

#visualise images
#obj.visualize_images(dir_path,nimages=3)
#Preprocess data
lung_df,train,labels=obj.preprocess(dir_path)
print(train)
print(labels)
# train and label datframe to csv
lung_df.to_csv("lung.csv",index=False)
#0train genarator, test genarator , validate genarator
tr_gen,tt_gen,va_gen=obj.generate_train_test_split(lung_df,train,labels)
def autoencoder_call():
    autoen=models.Autoencoder()
    autoencoder_model,train_gen,test_gen=autoen.anmodel()
    # Train the autoencoder using the train data generator
    autoencoder_model.fit(train_gen, epochs=50,validation_data=test_gen)
    autoencoder_model.save('autoencoder_saved.h5')
    # Evaluate the model on the test dataset
    loss = autoencoder_model.evaluate(train_gen)
    print("Test loss:", loss)
    model = load_model('autoencoder_saved.h5')
    nimages=10
    # Load a batch of images from the test generator
    test_images, _ = next(train_gen)
    # Predict images using the autoencoder
    predicted_images = autoencoder_model.predict(test_images)
    # Display original and predicted images side by side
    obj.display_images(test_images, predicted_images,5)

ms=models.DeepANN()

input_shape=(128,128,3)
mss=[]
'''
mss.append(ms.simple_ANN(input_shape=input_shape,optimizer="sgd"))
mss.append(ms.simple_ANN(input_shape=input_shape,optimizer="adam"))
mss.append(ms.simple_ANN(input_shape=input_shape,optimizer="rmsprop"))
#cnn
mss.append(ms.cnn_model(input_shape=input_shape,optimizer="adam"))
mss.append(ms.cnn_model(input_shape=input_shape,optimizer="spd"))
mss.append(ms.cnn_model(input_shape=input_shape,optimizer="rmsprop"))
#cnn-regularize
mss.append(ms.cnn_add_regularize())
mss.append(ms.cnn_vgg())  
#RNN
mss.append(ms.rnn(input_shape=input_shape,num_class=3))

'''
#call function for ---> AUTO ENCODER
#autoencoder_call()

mss.append(ms.LSTM_model(input_shape=input_shape,num_class=3))
models.compare_model(mss,tr_gen,va_gen,tt_gen,5)



