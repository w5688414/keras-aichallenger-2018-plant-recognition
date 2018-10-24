from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator  
import numpy as np
from tqdm import tqdm
import json
import os


def export_to_json(json_name,result):
    with open(json_name, 'w') as f:
        json.dump(result, f)
    print('write result json, num is %d' % len(result))
def format_results(generator,model):
    generator.reset()
    pred=model.predict_generator(generator,steps=2,verbose=1)
    predicted_class_indices=np.argmax(pred,axis=1)
    filenames=generator.filenames
    result = []
    for i in tqdm(range(len(filenames))):
        temp_dict = {}
        temp_dict['image_id'] = filenames[i].split('/')[-1]
        temp_dict['disease_class'] = int(predicted_class_indices[i])
        result.append(temp_dict)
    return result


## model 1
json_name1='inception_v4.json'
if(not os.path.exists(json_name1)):
    img_width, img_height = 229, 229
    test_datagen = ImageDataGenerator(1.0/255)
    generator=test_datagen.flow_from_directory(
                            '/home/eric/data/ai_challenger_plant_train_20170904/AgriculturalDisease_testA',
                            target_size=(img_width,img_height),
                            batch_size=1,
                            class_mode=None,
                            shuffle=False,
                            seed=42
                        )

    model=load_model('./trained_model/inception_v4/inception_v4.30-0.8103.hdf5')
    print(model.summary())
    result=format_results(generator,model)

    export_to_json(json_name1,result)


## model 2

json_name2='inception_resnet_v2.json'
if(not os.path.exists(json_name2)):
    img_width, img_height = 224, 224
    test_datagen = ImageDataGenerator(1.0/255)
    generator=test_datagen.flow_from_directory(
                            '/home/eric/data/ai_challenger_plant_train_20170904/AgriculturalDisease_testA',
                            target_size=(img_width,img_height),
                            batch_size=1,
                            class_mode=None,
                            shuffle=False,
                            seed=42
                        )
    model=load_model('./trained_model/inception_resnet_v2/inception_resnet_v2.15-0.8187.hdf5')
    result=format_results(generator,model)

    export_to_json(json_name2,result)

## model 3
json_name3='densenet.json'
if(not os.path.exists(json_name3)):
    img_width, img_height = 224, 224
    test_datagen = ImageDataGenerator(1.0/255)
    generator=test_datagen.flow_from_directory(
                            '/home/eric/data/ai_challenger_plant_train_20170904/AgriculturalDisease_testA',
                            target_size=(img_width,img_height),
                            batch_size=1,
                            class_mode=None,
                            shuffle=False,
                            seed=42
                        )
    from custom_layers import Scale
    model=load_model('./trained_model/densenet/densenet.17-0.8109.hdf5',custom_objects={'Scale':Scale})
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    print(model.summary())
    result=format_results(generator,model)


    export_to_json(json_name2,result)

with open(json_name1,'r') as js:
    text = json.loads(js.readline())
    print(text[0])
list_dict={}
for file in text:
    list_dict[file['image_id']]=[file['disease_class']]

with open(json_name2,'r') as js:
    text = json.loads(js.readline())
for file in text:
    list_dict[file['image_id']].append(file['disease_class'])

with open(json_name3,'r') as js:
    text = json.loads(js.readline())
for file in text:
    list_dict[file['image_id']].append(file['disease_class'])
## votes
result=[]
from collections import Counter
for (k,v) in  list_dict.items():
    tem_dict={}
    tem_dict['image_id']=k
    dict1=dict(Counter(v))
    sorted_dict=sorted(dict1.items(), key=lambda d: d[1],reverse=True)
    tem_dict['disease_class']=sorted_dict[0][0]
    result.append(tem_dict)

# final submit
with open('combine_submit.json', 'w') as f:
    json.dump(result, f)
    print('write result json, num is %d' % len(result)) 



