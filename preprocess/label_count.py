import os
import os.path as osp
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
'''
root = '/hdd2/wh/CVTT/JPEGImages'
img_list = os.listdir(root)
for img in img_list:
    f = open("trainval_aug.txt", "a")
    print(img.split('.jpg')[0], file=f)
    f.close()
'''

ids=[]
fo = open("trainval_aug.txt","r")
for line in fo.readlines():
    line = line.strip()
    ids.append(line)
info_path='/hdd2/wh/CVTT/Annotations'
annopath = osp.join(info_path, '%s.xml')
res = [0,0,0,0,0]
for item in ids:
    target = ET.parse(annopath % item).getroot()
    for obj in target.iter('object'):
        name = obj.find('name').text.lower().strip()
        if name.startswith('j') or name.startswith('w'):
            label = 0
        elif name.startswith('z') or name.startswith('p'):
            label = 1
        elif name.startswith('s') or name.startswith('i'):
            label = 2
        elif name.startswith('l'):
            label = 3
        elif name.startswith('d'):
            label = 4
        res[label] += 1
print(res)
