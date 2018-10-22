import json

label_path='/home/eric/data/ai_challenger_plant_train_20170904/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json'
data_dict = {}
with open(label_path, 'r') as f:
    label_list = json.load(f)
for image in label_list:
    data_dict[image['image_id']] = int(image['disease_class'])