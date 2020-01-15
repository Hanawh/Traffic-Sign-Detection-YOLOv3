import os.path as osp
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

ids=[]
fo = open("utils/data.txt","r")
for line in fo.readlines():
    line = line.strip()
    ids.append(line)
info_path='/home/wh/traffic_datasets/data/Annotations'
annopath = osp.join(info_path, '%s.xml')

bboxes = []
for item in ids:
    target = ET.parse(annopath % item).getroot()
    for obj in target.iter('object'):
        bbox = obj.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text)
            bndbox.append(cur_pt)
        width = bndbox[2]-bndbox[0]
        height = bndbox[3]-bndbox[1]
        bboxes.append([width, height])

bboxes = np.array(bboxes).reshape(-1, 2)

clf = KMeans(n_clusters=9)#.fit_predict(bboxes)
pred = clf.fit(bboxes)
plt.scatter(bboxes[:, 0], bboxes[:, 1], c=clf.labels_)
plt.show()
plt.savefig('kmeans.png')
print(clf.cluster_centers_)

