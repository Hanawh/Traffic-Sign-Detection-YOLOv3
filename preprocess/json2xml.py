import os
import json
import sys
from lxml import etree as ET
from xml.dom import minidom
# 创建好存储路径

def edit_xml(objects, id, dir):
    save_xml_path = os.path.join(dir, "%s.xml" % id)  # xml

    root = ET.Element("annotation")  
    folder = ET.SubElement(root, "folder")
    folder.text = "none"
    filename = ET.SubElement(root, "filename")
    filename.text = "none"
    source = ET.SubElement(root, "source")
    source.text = "201912"
    owner = ET.SubElement(root, "owner")
    owner.text = "HANA"
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(2048)
    height = ET.SubElement(size, "height")
    height.text = str(2048)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    for obj in objects:  #  
        object = ET.SubElement(root, "object")
        name = ET.SubElement(object, "name")  # number
        name.text = obj["category"]
        # meaning = ET.SubElement(object, "meaning")  # name
        # meaning.text = inf_value[0]
        pose = ET.SubElement(object, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(object, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(object, "difficult")
        difficult.text = "0"
        bndbox = ET.SubElement(object, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(obj["bbox"]["xmin"]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(obj["bbox"]["ymin"]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(obj["bbox"]["xmax"]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(obj["bbox"]["ymax"]))
    tree = ET.ElementTree(root)
    tree.write(save_xml_path, encoding="UTF-8", xml_declaration=True)
    root = ET.parse(save_xml_path) 
    file_lines = minidom.parseString(ET.tostring(root, encoding="Utf-8")).toprettyxml(
        indent="\t") 
    file_line = open(save_xml_path, "w", encoding="utf-8")  
    file_line.write(file_lines)
    file_line.close()
    
def  getDirId(dir):  # get the  id list  of id.png
    names = os.listdir(dir)
    ids = []
    for name in names:
        ids.append(name.split(".")[0])
    return ids  

filedir = "/hdd2/wh/Tsinghua/annotations.json"
annos = json.loads(open(filedir).read())
trainIds =  getDirId("/hdd2/wh/Tsinghua/train/")
testIds =  getDirId("/hdd2/wh/Tsinghua/test/")
ids = annos["imgs"].keys() #  all img ids in .json 

for id in ids:
    if len(annos["imgs"][id]["objects"]) > 0 and (id in trainIds):
        objects = annos["imgs"][id]["objects"]
        edit_xml(objects, id, dir = "/hdd2/wh/Tsinghua/xml/train")
        
    elif len(annos["imgs"][id]["objects"]) > 0 and (id in testIds):
        objects = annos["imgs"][id]["objects"]
        edit_xml(objects, id, dir = "/hdd2/wh/Tsinghua/xml/test")