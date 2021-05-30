import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing import image
from PIL import Image
import os


class Classifier:
    def __init__(self,exprmnt,classes):
        self.model=tf.keras.models.load_model(self.abspath(exprmnt+'.h5'))
        
        self.img_height, self.img_width = 128, 128
        self.classes = classes
    
    def abspath(self,filename):
        self.path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(self.path, filename)
           
    def predict_opencv_image(self, img):
        # convert the color from BGR to RGB then convert to PIL array
        cvt_image =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(cvt_image)
        # resize the array (image) then PIL image
        im_resized = im_pil.resize((self.img_width, self.img_height))
        img_array = image.img_to_array(im_resized)
        return self.predict(img_array)

    def predict(self,img_array=None,th=.5):
        if img_array is None:
            #sunflower_path = 'DB/mask_nomask/nomask/0_0_chenxiang_0006.jpg'
            sunflower_path = 'DB/mask_nomask/mask/0_0_0 copy 4.jpg'
            img = tf.keras.preprocessing.image.load_img(
                sunflower_path, target_size=(self.img_height, self.img_width)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)

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
    class_obj = MaskClassifier()
    
    #img = cv2.imread('DB/mask_nomask/mask/0_0_0 copy 4.jpg')
    img = cv2.imread('DB/mask_nomask/nomask/0_0_chenxiang_0006.jpg')
    
    classname,confidence = class_obj.predict_opencv_image (img)
    
    print(classname,confidence)
    
    
