import os 

from keras.preprocessing.image import ImageDataGenerator

# data pre-processing and augmentation
#initialising the data generators


dir_ = '/media/rtb7syl/New Volume/Projects-Workspace/Breast-Cancer-Classification-from-Histopathology-Images'
train_dir = os.path.join(dir_,"data/patch/vn_train_512_overlap")

val_dir = os.path.join(dir_,"data/patch/vn_validation_512_overlap")
#test_dir = "../data/test_local"
    


train_datagen = ImageDataGenerator(
    
                    rescale=1./255,
                    rotation_range=180,
                    horizontal_flip=True,
                    vertical_flip=True
                )


val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    
                                train_dir,
                                target_size=(450, 450),
                                batch_size=8,
                                class_mode="categorical"
                            )


validation_generator = val_datagen.flow_from_directory(
    
                             val_dir,
                             target_size=(450, 450),
                             batch_size=8,
                             class_mode="categorical")

