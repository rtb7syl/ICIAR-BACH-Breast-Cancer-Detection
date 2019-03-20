#import staintools
import os

from sklearn.feature_extraction import image

from skimage import io


'''
#STAIN NORMALIZATION

def normalize(impath):
    
    # Read data
    to_augment = staintools.read_image(impath)

    # Standardize brightness (This step is optional but can improve the tissue mask calculation)
    standardizer = staintools.BrightnessStandardizer()
    to_augment = standardizer.transform(to_augment)

    # Stain augment
    augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)
    augmentor.fit(to_augment)
    augmented_image = augmentor.pop()
    
    for _ in range(100):
        augmented_image = augmentor.pop()
        augmented_images.append(augmented_image)
        
    
    
    return augmented_image


def normalize_img_and_write_to_disk(path,target_path):
    
    class_dirs = ['Benign','InSitu','Invasive','Normal']
    print(class_dirs)
    for class_dir in class_dirs:
        
        class_dir_abs_path = path+'/'+class_dir
        class_target_path = target_path+'/'+class_dir
        
        print(class_dir_abs_path,class_target_path)
        class_imgs = os.listdir(class_dir_abs_path)
        
        for class_img in class_imgs:
            
            img_identifier = class_img[-4:]
            print(img_identifier)
            
            if (img_identifier == '.tif' or img_identifier == 'tiff'):
            
                impath = class_dir_abs_path+'/'+class_img
                im_target_path = class_target_path+'/'+class_img
                print(impath,im_target_path)

                transformed_img = normalize(impath)

                print(impath+' normalized')

                cv2.imwrite(im_target_path,transformed_img)
                print('img written')
            


'''

# EXTRACTING PATCHES

#after stain normalizing the images,extract 512*512 patches from each image with stride 256
#and save it to the corresponding label directory



def extract_patches_for_each_class(class_dir_path,patch_dims):
    #patch_dims is the size of the patch being extracted ie 512,256 etc
    
    img_names = os.listdir(class_dir_path)
    
    for img_name in img_names:
        
        impath = os.path.join(class_dir_path,img_name)
        
        print("   [INFO] Processing img: ",impath,'\n')
        
        img = io.imread(impath)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        '''
        #patches = image.extract_patches_2d(img, (patch_dims, patch_dims))
        #print(patches.shape)
        
        #n_patches = patches.shape[0]
        
        for i in range(n_patches):
            
            print("     [INFO] Processing patch: ",i,'\n')
        
            yield patches[i]
        '''
        
        x_range = ((2048//patch_dims)*2)-1
        y_range = ((1536//patch_dims)*2)-1
        
        stride = patch_dims//2
        
        print(x_range,y_range)
        
        for x in range(x_range):
            
            for y in range(y_range):
                
                print('x,y',x,y)
                patch = img[stride*y:stride*(y+2),stride*x:stride*(x+2)]
                fname = img_name.split('.')[0]+'('+str(x)+','+str(y)+')'+'.tif'
                
                yield (patch,fname)
                
                

def write_patch_to_disk(patch,target_path):
    
    #target_path = target_path.split('.')[1]+'.jpg'
    #print(target_path)
    
    io.imsave(target_path,patch)
    
    
def extract_patches_from_every_class_and_write_to_disk(path,to_path):
    
    class_dirs = ['Benign','InSitu','Invasive','Normal']
    
    for class_dir in class_dirs:
        
        print("[INFO] Processing class: ",class_dir,'\n')
        
        class_dir_path = os.path.join(path,class_dir)
        
        for patch,img_name in extract_patches_for_each_class(class_dir_path,512):
            
            #img_name = img_name.split('.')[0]+'.jpg'
            
            
            target_path = os.path.join(to_path,class_dir,img_name)
            
            print(target_path)
            write_patch_to_disk(patch,target_path)
            
            print("[INFO] Written img to: ",target_path,'\n')


def extract_patches(img,patch_dims):

    #generator to yield patches of an image img
    #patch_dims is the size of the patch being extracted ie 512,256 etc
    

        
    x_range = ((2048//patch_dims)*2)-1
    y_range = ((1536//patch_dims)*2)-1
    
    stride = patch_dims//2
    
    print(x_range,y_range)

    #patch index
    i = 0
    
    for x in range(x_range):
        
        for y in range(y_range):
            
            print('x,y',x,y)
            patch = img[stride*y:stride*(y+2),stride*x:stride*(x+2)]

            i += 1

            print('No. ' + str(i) + ' patch')
            
            yield (patch,i)