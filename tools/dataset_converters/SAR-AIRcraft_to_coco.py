# 运行本文件，将雷达学报 SAR-aircraft 1.0的标注格式从VOC改成COCO格式
# COCO格式的标注文件存储在'coco_folder'中
# python tools/dataset_converters/SAR-AIRcraft_to_coco.py

from globox import AnnotationSet
import os
import shutil

txt_foler = 'data/SAR-AIRcraft-1.0/ImageSets/Main'
txt_files = ['test.txt', 'train.txt', 'trainval.txt', 'val.txt']
extension = '.xml'
xml_folder = 'data/SAR-AIRcraft-1.0/Annotations'
tmp_folder = 'data/SAR-AIRcraft-1.0/tmp'
coco_foler = '/root/mmdetection/data/SAR-AIRcraft-1.0/COCOAnnotations'

for txt_file in txt_files:

    # 从txt中读取xml的文件名
    shutil.rmtree(tmp_folder)
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    json_file = txt_file.split('.')[0]
    txt_file = os.path.join(txt_foler, txt_file)
    with open(txt_file, 'r') as file:
        xml_files = file.read().splitlines()

    # 将xml注释复制到临时文件夹
    for xml_file in xml_files:
        xml_file_path = os.path.join(xml_folder, xml_file + extension)
        shutil.copy(xml_file_path, tmp_folder)
    
    # 解析VOC目标检测标注
    pascal = AnnotationSet.from_pascal_voc(folder=tmp_folder)
    pascal.show_stats()

    
    imageid_to_id = {im: int(im.split('.')[0]) for im in pascal.image_ids}
    # {'A220': 0, 'A320/321': 1, 'A330': 2, 'ARJ21': 3, 'Boeing737': 4, 'Boeing787': 5, 'other': 6}
    label_to_id = {l: i for i, l in enumerate(sorted(pascal._labels()))}
    
    pascal.save_coco(path=os.path.join(coco_foler, json_file), imageid_to_id = imageid_to_id, label_to_id=label_to_id)
