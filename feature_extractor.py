# This module contains functions to extract feature vectors from the trained CNN models

from skimage import io


from models import InceptionModel
from data_generator import validation_generator

import os
import numpy as np


def extract_feature_vector(img,model,inception_model,out_path):

    intermediate_output = inception_model.extract_layer_output(img,model)

    np.savetxt(out_path,intermediate_output)



def extract_feature_vector_of_all_imgs(dir_,out,weights_path):

    inception_model = InceptionModel()
    
    model = inception_model.create()

    model.load_weights(weights_path)

    print('[INFO].....Model weights loaded......')

    train_imgs_dir = os.path.join(dir_,'vn_train_512_overlap')

    val_imgs_dir = os.path.join(dir_,'vn_validation_512_overlap')

    imgs_dirs = [val_imgs_dir]
    imgs_dirs = [train_imgs_dir,val_imgs_dir]

    imgs_classes = ['Benign','InSitu','Invasive','Normal']

    for i in range(2):

        if (i == 0):

            out_dir = os.path.join(out,'train')

            print('[INFO].....Processing train imgs......')

        else:

            out_dir = os.path.join(out,'val')

            print('[INFO].....Processing val imgs......')

        imgs_dir = imgs_dirs[i]

        for imgs_class in imgs_classes:

            print('[INFO].....Processing ',imgs_class,' ......')

            class_dir = os.path.join(imgs_dir,imgs_class)
            out_class_dir = os.path.join(out_dir,imgs_class)

            imnames = os.listdir(class_dir)

            for imname in imnames:

                impath = os.path.join(class_dir,imname)

                print('[INFO].....Processing ',impath,' ......')


                out_fname = imname[:-4] + '.csv'
                out_path = os.path.join(out_class_dir,out_fname)
                print('[INFO].....out_path',out_path,' ......')

                img = io.imread(impath)

                if img is None:

                    raise ValueError('Inappropriate value of impath')

                extract_feature_vector(img,model,inception_model,out_path)












    




if __name__ == "__main__":

    '''

    impath = '/media/rtb7syl/New Volume/Projects-Workspace/Breast-Cancer-Classification-from-Histopathology-Images/data/stain_norm_imgs_validation/Invasive'

    impath = os.path.join(impath,'iv003.tif')
    weights_path = 'inceptionv3_model_weights_checkpoint/weights-improvement-post_6-03-0.78.h5'
    out_path = 'yy.csv'
    img = io.imread(impath)

    if img is None:

        raise ValueError('Inappropriate value of impath')


    extract_feature_vector(img,weights_path,out_path)

    '''
    dir_ = '/media/rtb7syl/New Volume/Projects-Workspace/Breast-Cancer-Classification-from-Histopathology-Images/data/patch'
    out_dir = 'feature_vectors/Inceptionv3'
    weights_path = 'inceptionv3_model_weights_checkpoint/weights-improvement-post_6-03-0.78.h5'
    extract_feature_vector_of_all_imgs(dir_,out_dir,weights_path)