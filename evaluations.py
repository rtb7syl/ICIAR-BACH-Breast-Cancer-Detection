
from skimage import io

from utils import extract_patches
from models import InceptionModel
from data_generator import validation_generator

import os
import random




def predict_one_img(img,
                
                weights_path,
                inception_model,
                model,
                labels = {'Benign': 0, 'InSitu': 1, 'Invasive': 2, 'Normal': 3}

            ):

    #predicts the class label of one 2048*1536 img
    #labels is a the class labels dict
    #return the label 0,1,2 or 3

    label_vals = list(labels.values())

    print('label_vals',label_vals)



    #list to hold predicted labels and confidence scores of the patches
    patch_pred_labels = []
    patch_pred_confs = []



    for patch,index in extract_patches(img,512):

        #print('[INFO]....Processing patch no. '+str(index))

        #model = InceptionModel()

        label,confidence = inception_model.predict(patch,model,weights_path)

        patch_pred_labels.append(label)
        patch_pred_confs.append(confidence)

        #print('patch_pred_labels,patch_pred_confs',patch_pred_labels,patch_pred_confs)



    
    #Majority Voting Decision over all the patches 
    #to predict class label of entire image

    #predict label with max occurences in patch_pred_labels
    #if tie, choose label with max total confidence
    #if tie again , choose any of those labels at random

    patches_pred_label_freqs = {}

    for label in label_vals:

        label_freq = patch_pred_labels.count(label)

        #print('label '+str(label)+' has a freq of ',label_freq)

        patches_pred_label_freqs[str(label)] = label_freq

        #print('patches_pred_label_freqs',patches_pred_label_freqs)

    #print(patches_pred_label_freqs)

    max_freq = max(patches_pred_label_freqs.values())

    #print('max_freq',max_freq)

    #list of labels with max_freq
    argmax_freqs = [int(k) for k, v in patches_pred_label_freqs.items() if v == max_freq]
    
    #print('list of labels with max_freq',argmax_freqs)


    if (len(argmax_freqs) == 1):

        print('Only one class majority')

        print('Predicted Label : ',argmax_freqs[0])

        return argmax_freqs[0]


    elif (len(argmax_freqs) > 1):

        #if tie, choose label with max total confidence

        print('Multiple classes with same majority')

        argmax_confs_dict = {}

        for argmax_freq in argmax_freqs:

            total_confidence = 0
            total_instances = 0

            #print('Current label',argmax_freq)
            
        
            for label,confidence in zip(patch_pred_labels,patch_pred_confs):

                #print('label',label)

                if (label == argmax_freq):

                    total_confidence += confidence
                    total_instances += 1

                    #print('total_confidence',total_confidence)
                    #print('total_instances',total_instances)
                    

            mean_confidence = total_confidence/max_freq
            #print('mean_confidence',mean_confidence)

            argmax_confs_dict[str(argmax_freq)] = mean_confidence

            #print('argmax_confs_dict',argmax_confs_dict)

        max_conf = max(argmax_confs_dict.values())
        #print('max_conf',max_conf)

        #list of labels with max_conf
        argmax_confs = [int(k) for k, v in argmax_confs_dict.items() if v == max_conf]

        #print('argmax_confs',argmax_confs)

        if (len(argmax_confs) == 1):

            print('Predicted Label : ',argmax_confs[0])

            return argmax_confs[0]

        elif (len(argmax_confs) > 1):

            #if tie again,choose any of those labels at random

            op_label = random.choice(argmax_confs)
            print('Predicted Label : ',op_label)

            return op_label






def evaluate_all_imgs(dir_,
                    
                weights_path,
                
                labels = {'Benign': 0, 'InSitu': 1, 'Invasive': 2, 'Normal': 3}

                ):

    #returns the image level accuracy of the entire validation set

    inception_model = InceptionModel()
    
    model = inception_model.create()

    model.load_weights(weights_path)

    print('[INFO].....Model weights loaded......')


    #if predicted label of ith image in the validation set 
    #matches with its ground-truth label
    #ith element in this list is 1 else 0  
    imgs_pred_scores = []

    img_idx = 0

    class_dirs = list(labels.keys())
    
    for class_dir in class_dirs:

        print('[INFO]....Currently processing class ',class_dir)

        truth_label = labels[class_dir]

        class_dir_path = os.path.join(dir_,class_dir)

        imnames = os.listdir(class_dir_path+'/')

        for imname in imnames:

            impath = os.path.join(class_dir_path,imname)

            print('[INFO]....Currently processing img ',imname)

            img = io.imread(impath)

            if img is None:

                raise ValueError('Inappropriate value of impath')

            predicted_label = predict_one_img(img,weights_path,inception_model,model,labels)

            img_pred_score = 1 if (predicted_label == truth_label) else 0

            print('img_pred_score',img_pred_score)

            imgs_pred_scores.append(img_pred_score)

            print('imgs_pred_scores',imgs_pred_scores)

            img_idx += 1

            print('[INFO]....'+str(img_idx)+' images processed')

            acc = sum(imgs_pred_scores)/len(imgs_pred_scores)

            print('[INFO]....Image level accuracy = ',str(acc))



def evaluate_patches(weights_path):

    inception_model = InceptionModel()
    
    model = inception_model.init(weights_path)

    loss,acc = model.evaluate_generator(generator=validation_generator,steps=348)

    print('loss,acc',loss,acc)








if __name__ == "__main__":

    '''

    #dir_ = '/media/rtb7syl/New Volume/Projects-Workspace/Breast-Cancer-Classification-from-Histopathology-Images'

    impath = '/media/rtb7syl/New Volume/Projects-Workspace/Breast-Cancer-Classification-from-Histopathology-Images/data/stain_norm_imgs_validation/Invasive'

    impath = os.path.join(impath,'iv003.tif')

    img = io.imread(impath)

    if img is None:

        raise ValueError('Inappropriate value of impath')

    predicted_label = predict_one_img(img,'inceptionv3_model_weights_checkpoint/weights-improvement-post_6-03-0.78.h5')

    print('Ground Truth Label = 2')
    print('Predicted Label ',predicted_label)

    '''

    #evaluate_patches('inceptionv3_model_weights_checkpoint/weights-improvement-post_6-03-0.78.h5')

    dir_ = '/media/rtb7syl/New Volume/Projects-Workspace/Breast-Cancer-Classification-from-Histopathology-Images/data/stain_norm_imgs_validation'
    weights_path = 'inceptionv3_model_weights_checkpoint/weights-improvement-post_6-03-0.78.h5'

    evaluate_all_imgs(dir_,weights_path)



    




































        

    































