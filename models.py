import os
import pickle

import numpy as np
from skimage import transform


from keras.applications import inception_v3

from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from data_generator import train_generator,validation_generator




class InceptionModel():

    # inceptionv3 model

    def __init__(self,ip_shape=(450,450,3)):

        self.ip_shape = ip_shape

        self.conv_base = inception_v3.InceptionV3(include_top=False,weights='imagenet',input_shape=self.ip_shape,pooling=None)

        for i in range(4):
            
            layer = self.conv_base.layers.pop()


        print('labels from gen',validation_generator.class_indices,train_generator.class_indices)
            
            

        #print(self.conv_base.summary())


    def create(self):

        print('[INFO].....Model started building......')
        model = models.Sequential()
        model.add(self.conv_base)


        model.add(layers.GlobalAveragePooling2D(data_format='channels_last'))
        model.add(layers.Dense(1024, activation='relu'))

        #model.add(layers.Dense(512, activation='relu'))
        #model.add(layers.Dropout(0.5))
        #model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))

        print('[INFO].....Model built......')

        return model


    def init(self,weights_path = 'inceptionv3_model_weights_checkpoint/weights-improvement-06-0.78.h5'):

        model = self.create()


        model.compile(loss='categorical_crossentropy',
                    
                    optimizer= optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=True),
                    
                    metrics=[metrics.categorical_accuracy])

        
        model.load_weights(weights_path)

        print('[INFO].....Model initialised......')

        #print('[INFO].....Model summary......')
        #print(model.summary())

        

        return model

    
    def train(self,weights_path,
                checkpoint_path="inceptionv3_model_weights_checkpoint/weights-improvement-post_6-{epoch:02d}-{val_acc:.2f}.h5"
               ):


        model = self.init(weights_path=weights_path)

        print('[INFO].....Model started training......')

        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)

        #reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, verbose=1, mode='auto', cooldown=1, min_lr=0.0001)


        # Train the model 
        history = model.fit_generator(

            train_generator,
            steps_per_epoch = 1394,
            epochs = 40,
            validation_data = validation_generator,
            validation_steps=348,
            callbacks = [checkpoint]
            
        )

        # dump the training history data in a pkl 
        with open('inceptionv3_sgdn_lr0_0001_ep80_history.pickle', 'wb') as handle:
            pickle.dump(history.history, handle)
            


        print('[INFO].....Model finished training......')

        print('[INFO].....Model history......')

        print(history.history)



    def predict(self,img,model,weights_path):

        #predict class label either 0,1,2,3 and confidence score of the prediction 
        #of an input img (of shape (M,N,3)) after applying some transformations on it
        #the created model with loaded weights must be passed as an argument


        # resize and expand dims from 3 to 4, and normalize img

        img = np.array(img).astype('float32')/255

        img = transform.resize(img, self.ip_shape)

        img = np.expand_dims(img, axis=0)
        

        #model = self.create()
        
        #model.load_weights(weights_path)

        label = model.predict_classes(img)

        confidence = model.predict_proba(img)

        print('label,confidence',label,confidence)

        label = label[0]
        confidence = confidence[0][label]

        print('label,confidence',label,confidence)

        return label,confidence


















        

     