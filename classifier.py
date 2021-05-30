CUDA_VISIBLE_DEVICES=""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing import image
from PIL import Image
import os

import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#  tf.config.experimental.set_memory_growth(gpu, True)
  
tf.config.set_visible_devices([], 'GPU')

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
    

class Classifier:
    def __init__(self,exprmnt,classes):
        self.model=tf.keras.models.load_model(exprmnt+'.h5')
        
        self.img_height, self.img_width = 128, 128
        self.classes = classes
    
           
    def predict_opencv_image(self, img):
        # convert the color from BGR to RGB then convert to PIL array
        cvt_image =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(cvt_image)
        # resize the array (image) then PIL image
        im_resized = im_pil.resize((self.img_width, self.img_height))
        img_array = image.img_to_array(im_resized)
        return self.predict(img_array)

    def predict(self,img_array=None,th=.5):

        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions)
        print( score)
        
        classid = score[0].numpy().argmax()
        confidence = score[0].numpy()[classid]
        
        print(self.classes[classid])
        #import pdb;pdb.set_trace()
        return self.classes[classid], confidence
        
if __name__ == '__main__':
    class_obj = Classifier(exprmnt = 'warning',classes=['other', 'warning_for_children_and_minors'])
    
    #img = cv2.imread('DB/mask_nomask/mask/0_0_0 copy 4.jpg')
    img = cv2.imread('DB/warning/warning_for_children_and_minors/1_tensor(0) (copy).png')
    
    classname,confidence = class_obj.predict_opencv_image (img)
    
    print(classname,confidence)
    
    
