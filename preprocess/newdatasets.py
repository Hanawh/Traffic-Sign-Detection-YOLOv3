import cv2
from shutil import copyfile
import os.path as osp
'''
CV_dir = 'CV_l.txt'
fo = open(CV_dir,"r")
lines = fo.readlines()

for line in lines:
    line = line.strip()
    ab_dir = '/hdd2/wh/Tsinghua'
    img_dir = osp.join(ab_dir,'train',line+'.jpg')
    ann_dir = osp.join(ab_dir,'xml/train',line+'.xml')
    
    load_dir = '/hdd2/wh/CVTT'
    load_img = osp.join(load_dir,'JPEGImages',line+'.jpg')
    load_ann = osp.join(load_dir,'Annotations',line+'.xml')
    
    # copy xml
    copyfile(ann_dir, load_ann)
    copyfile(img_dir, load_img)
'''

CV_dir = 'CV_l.txt'
fo = open(CV_dir,"r")
for line in fo.readlines():
    line = line.strip()
    ab_dir = '/hdd2/wh/traffic_datasets/data'
    img_dir = osp.join(ab_dir,'JPEGImages',line+'.jpg')
    ann_dir = osp.join(ab_dir,'Annotations',line+'.xml')
    
    load_dir = '/hdd2/wh/CVTT'
    load_img = osp.join(load_dir,'JPEGImages',line+'_aug.jpg')
    load_ann = osp.join(load_dir,'Annotations',line+'_aug.xml')
    
    # copy xml
    copyfile(ann_dir, load_ann)
    copyfile(img_dir, load_img)
    

