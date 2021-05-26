import os
import numpy as np
import shutil 

def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if 1: #(fabric case)
    src_images_path = 'data/dataset_fabric_defect'
    img_ext = ['.tif']
    dst_image_p = 'data/allimages'
    dst_label_p = 'data/alllabels'
    
    mkdir(dst_image_p)
    mkdir(dst_label_p)
    cnt=0

    # Cleanup mixed images and labels 
    
    for dirname, _, filenames in os.walk(src_images_path):
        for filename in filenames:
            filepath = os.path.join(dirname, filename)
            #import pdb;pdb.set_trace()
            
            if os.path.splitext(filepath)[1] in img_ext:
                img_path   = filepath
                annot_path = os.path.splitext(img_path)[0]+'.xml'
                
                if os.path.isfile(annot_path):
                
                    dst_image = os.path.splitext(filename)[0]+'_'+str(cnt).zfill(3)+'.tif'
                    dst_label = os.path.splitext(filename)[0]+'_'+str(cnt).zfill(3)+'.xml'
                    
                    dst_image = os.path.join(dst_image_p, dst_image)
                    dst_label = os.path.join(dst_label_p, dst_label)
                    
                    shutil.copy(img_path, dst_image) 
                    shutil.copy(annot_path, dst_label) 
                    print('copied :' ,dst_image)
                    cnt+=1
                    

if 0:
    src_images_path = 'data/dataset_fabric_defect'
    img_ext = ['.tif']

    # Cleanup
    for dirname, _, filenames in os.walk(src_images_path):
        for filename in filenames:
            filepath = os.path.join(dirname, filename)
            import pdb;pdb.set_trace()
            
            if os.path.splitext(filepath)[1] in img_ext:
                
                
                annot_path = os.path.splitext(img_path)[0]+'.xml'
                head,tail1 = os.path.split(annot_path)
                head,tail2 = os.path.split(head)
                annot_path = os.path.join(head, 'annotations',tail1)
                if not os.path.isfile(annot_path):
                    os.remove(img_path)
                    print('removed:' ,img_path)
                    
                
                
