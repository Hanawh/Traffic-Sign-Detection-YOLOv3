import os
import os.path as osp
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

'''
train_dir = '/hdd2/wh/Tsinghua/xml/train'
annopath = osp.join(train_dir, '%s')
train_list = os.listdir(train_dir)
for img_id in train_list:
    anno = ET.parse(annopath % img_id).getroot()
    for obj in anno.iter('object'):
        name = obj.find('name').text.strip()
        if name.startswith('w'):
            f = open("TT_j0.txt", "a")
            print(img_id.split('.xml')[0], file=f)
            f.close()
        if name.startswith('i'):
            f = open("TT_s2.txt", "a")
            print(img_id.split('.xml')[0], file=f)
            f.close()
'''
train_dir = '/hdd2/wh/traffic_datasets/data/Annotations'
annopath = osp.join(train_dir, '%s.xml')

ids = []
fo = open("trainval.txt","r")
for line in fo.readlines():
    line = line.strip()
    ids.append(line)

for img_id in ids:
    anno = ET.parse(annopath % img_id).getroot()
    for obj in anno.iter('object'):
        name = obj.find('name').text.strip()
        if name.startswith('l'):
            f = open("CV_l.txt", "a")
            print(img_id, file=f)
            f.close()