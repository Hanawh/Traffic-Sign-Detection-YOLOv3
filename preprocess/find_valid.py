import os
import os.path as osp
import torch
import sys
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
info_path='/home/wh/traffic_datasets/data/Annotations'
image_path='/home/wh/traffic_datasets/data/JPEGImages'
info = os.listdir(info_path)

for item in info:
    temp = osp.join(info_path,item)
    target = ET.parse(temp).getroot()
    res = []
    for obj in target.iter('object'):
        name = obj.find('name').text.strip()
        bbox = obj.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            # scale height or width
            bndbox.append(cur_pt)
        res += [bndbox]
    boxes = torch.from_numpy(np.array(res).reshape(-1, 4))
    if(boxes.shape[0] != 0):
        n = item.replace('.xml','.jpg')
        if osp.isfile(osp.join(image_path, n)): #是否存在标注文件
           f = open("trainval.txt", "a")
           print(n.split('.jpg')[0], file=f)
           f.close()




