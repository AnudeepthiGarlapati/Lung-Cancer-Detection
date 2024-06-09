import os
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class preprocess_data:
    def display_images(self,original_images, predicted_images, num_images=5):
        fig, axs = plt.subplots(2, num_images, figsize=(10, 10))
        for i in range(num_images):
            #   # Original image
            axs[0, i].imshow(original_images[i])
            axs[0, i].set_title('Original')
            axs[0, i].axis('off')
            # Predicted image
            axs[1, i].imshow(predicted_images[i])
            axs[1, i].set_title('Predicted')
            axs[1, i].axis('off')
        plt.tight_layout()
        plt.savefig('static/images/compare.jpg')
        #plt.show(block=True)


    #write method to visualize images
    def visualize_images(self,dir_path,nimages):
        fig, axs = plt.subplots(3,3, figsize = (10, 10))  #9 images
        dpath=dir_path
        count=0
        for i in os.listdir(dpath):
            #get the list of images in a given class
            train_class=os.listdir(os.path.join(dpath,i))
            #plot the images
            for j in range(nimages):
                img=os.path.join(dpath,i,train_class[j])
                img=cv2.imread(img)
                axs[count][j].title.set_text(i)
                axs[count][j].imshow(img)
            count+=1
        fig.tight_layout()
        plt.show(block=True)

    #write method to preprocess the data
    def preprocess(self, dir_path):
        dpath=dir_path
        #count the number of images in the dataset
        train=[]
        labels=[]
        for i in os.listdir(dpath):
            #get the list of images in a given class
            train_class=os.listdir(os.path.join(dpath,i))
            for j in train_class:
                img=os.path.join(dpath,i,j)
                train.append(img)
                labels.append(i)
        print("number of images:{}\n".format(len(train)))
        print("number of image labels:{}\n".format(len(labels)))
        leaf_df=pd.DataFrame({'Image':train, 'Labels':labels})
        print(leaf_df)
        return leaf_df,train,labels

    #skill4   ==>Image data generator
    def generate_train_test_split(self, retina_df, train, labels):
        train_df, test_df = train_test_split(retina_df, test_size=0.2)

        train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, validation_split=0.2)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(128,128),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=12,
            subset='training'
        )

        validate_generator = train_datagen.flow_from_dataframe(
            train_df,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(128,128),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=12,
            subset='validation'
        )

        test_generator = test_datagen.flow_from_dataframe(
            test_df,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(128,128),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=12
        )

        print(f"Train image shape: {train_df.shape}")
        print(f"Test image shape: {test_df.shape}")

        return train_generator, test_generator, validate_generator

