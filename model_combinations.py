from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator  
import numpy as np
from tqdm import tqdm
import json
import os


def export_to_json(json_name,result):
    with open(json_name, 'w') as f:
        json.dump(result, f)
    print('write %s, num is %d' % (json_name,len(result)))
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

test_dir='/home/eric/data/plant/ai_challenger_pdr2018_testa_20181023/AgriculturalDisease_testA'

## model 1
json_name1='inception_v4.json'
if(not os.path.exists(json_name1)):
    img_width, img_height = 229, 229
    test_datagen = ImageDataGenerator(1.0/255)
    generator=test_datagen.flow_from_directory(
                            test_dir,
                            target_size=(img_width,img_height),
                            batch_size=1,
                            class_mode=None,
                            shuffle=False,
                            seed=42
                        )

    model=load_model('./trained_model/inception_v4/inception_v4.44-0.8700.hdf5')
    # print(model.summary())
    result=format_results(generator,model)

    export_to_json(json_name1,result)

with open(json_name1,'r') as js:
    text = json.loads(js.readline())
    # print(text[0])
list_dict={}
for file in text:
    list_dict[file['image_id']]=[file['disease_class']]

## model 2
json_name2='inception_resnet_v2.json'
if(not os.path.exists(json_name2)):
    img_width, img_height = 224, 224
    test_datagen = ImageDataGenerator(1.0/255)
    generator=test_datagen.flow_from_directory(
                            test_dir,
                            target_size=(img_width,img_height),
                            batch_size=1,
                            class_mode=None,
                            shuffle=False,
                            seed=42
                        )
    model=load_model('./trained_model/inception_resnet_v2/inception_resnet_v2.12-0.8244.hdf5')
    result=format_results(generator,model)

    export_to_json(json_name2,result)

with open(json_name2,'r') as js:
    text = json.loads(js.readline())
for file in text:
    list_dict[file['image_id']].append(file['disease_class'])


## model 3
# json_name3='densenet.json'
# if(not os.path.exists(json_name3)):
#     img_width, img_height = 224, 224
#     test_datagen = ImageDataGenerator(1.0/255)
#     generator=test_datagen.flow_from_directory(
#                             test_dir,
#                             target_size=(img_width,img_height),
#                             batch_size=1,
#                             class_mode=None,
#                             shuffle=False,
#                             seed=42
#                         )
#     from custom_layers import Scale
#     model=load_model('./trained_model/densenet/densenet.17-0.8109.hdf5',custom_objects={'Scale':Scale})
#     model.compile(loss='categorical_crossentropy',
#             optimizer='adam',
#             metrics=['accuracy'])
#     # print(model.summary())
#     result=format_results(generator,model)
#     export_to_json(json_name3,result)

# with open(json_name3,'r') as js:
#     text = json.loads(js.readline())
# for file in text:
#     list_dict[file['image_id']].append(file['disease_class'])

## model 4
json_name4='resnet50.json'
if(not os.path.exists(json_name4)):
    img_width, img_height = 265, 265
    test_datagen = ImageDataGenerator(1.0/255)
    generator=test_datagen.flow_from_directory(
                            test_dir,
                            target_size=(img_width,img_height),
                            batch_size=1,
                            class_mode=None,
                            shuffle=False,
                            seed=42
                        )
    model=load_model('./trained_model/resnet50/resnet50.27-0.8720.hdf5')
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    # print(model.summary())
    result=format_results(generator,model)

    export_to_json(json_name4,result)

with open(json_name4,'r') as js:
    text = json.loads(js.readline())
for file in text:
    list_dict[file['image_id']].append(file['disease_class'])

## model 5
# json_name5='mobilenet_v2.json'
# if(not os.path.exists(json_name5)):
#     img_width, img_height = 229, 229
#     test_datagen = ImageDataGenerator(1.0/255)
#     generator=test_datagen.flow_from_directory(
#                             test_dir,
#                             target_size=(img_width,img_height),
#                             batch_size=1,
#                             class_mode=None,
#                             shuffle=False,
#                             seed=42
#                         )
#     from mobilenet_v2 import relu6
#     from mobilenet_v2 import DepthwiseConv2D
#     model=load_model('./trained_model/mobilenet_v2/mobilenet_v2.43-0.8674.hdf5',custom_objects={'relu6':relu6,'DepthwiseConv2D':DepthwiseConv2D})

#     model.compile(loss='categorical_crossentropy',
#             optimizer='adam',
#             metrics=['accuracy'])
#     # print(model.summary())
#     result=format_results(generator,model)
#     export_to_json(json_name5,result)

# with open(json_name5,'r') as js:
#     text = json.loads(js.readline())
# for file in text:
#     list_dict[file['image_id']].append(file['disease_class'])

## model 6
# json_name6='shufflenet_v2.json'
# if(not os.path.exists(json_name6)):
#     img_width, img_height = 229, 229
#     test_datagen = ImageDataGenerator(1.0/255)
#     generator=test_datagen.flow_from_directory(
#                             test_dir,
#                             target_size=(img_width,img_height),
#                             batch_size=1,
#                             class_mode=None,
#                             shuffle=False,
#                             seed=42
#                         )
#     charset_size=61
#     from shufflenetv2 import ShuffleNetV2
#     model=ShuffleNetV2.ShuffleNetV2(input_shape=(img_width,img_height,3),classes=charset_size,weights='./trained_model/shufflenet_v2/shufflenet_v2.26-0.8676.hdf5')

#     model.compile(loss='categorical_crossentropy',
#             optimizer='adam',
#             metrics=['accuracy'])
#     # print(model.summary())
#     result=format_results(generator,model)
#     export_to_json(json_name6,result)


# with open(json_name6,'r') as js:
#     text = json.loads(js.readline())
# for file in text:
#     list_dict[file['image_id']].append(file['disease_class'])

## model 7
json_name='inception_v3.json'
if(not os.path.exists(json_name)):
    img_width, img_height = 224, 224
    test_datagen = ImageDataGenerator(1.0/255)
    generator=test_datagen.flow_from_directory(
                            test_dir,
                            target_size=(img_width,img_height),
                            batch_size=1,
                            class_mode=None,
                            shuffle=False,
                            seed=42
                        )
    model=load_model('./trained_model/inception_v3/inception_v3.43-0.8747.hdf5')
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    # print(model.summary())
    result=format_results(generator,model)

    export_to_json(json_name,result)

with open(json_name,'r') as js:
    text = json.loads(js.readline())
for file in text:
    list_dict[file['image_id']].append(file['disease_class'])

## model 8
json_name='seresnet50.json'
if(not os.path.exists(json_name)):
    img_width, img_height = 224, 224
    test_datagen = ImageDataGenerator(1.0/255)
    generator=test_datagen.flow_from_directory(
                            test_dir,
                            target_size=(img_width,img_height),
                            batch_size=1,
                            class_mode=None,
                            shuffle=False,
                            seed=42
                        )
    model=load_model('./trained_model/seresnet50/seresnet50.43-0.8795.hdf5')
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    # print(model.summary())
    result=format_results(generator,model)

    export_to_json(json_name,result)

with open(json_name,'r') as js:
    text = json.loads(js.readline())
for file in text:
    list_dict[file['image_id']].append(file['disease_class'])


## model 9
json_name='resnet_attention_56.json'
if(not os.path.exists(json_name)):
    img_width, img_height = 224, 224
    test_datagen = ImageDataGenerator(1.0/255)
    generator=test_datagen.flow_from_directory(
                            test_dir,
                            target_size=(img_width,img_height),
                            batch_size=1,
                            class_mode=None,
                            shuffle=False,
                            seed=42
                        )
    from resnet_attention_56 import Resnet_Attention_56
    charset_size=61
    model=Resnet_Attention_56.Resnet_Attention_56(input_shape=(img_width,img_height,3),classes=charset_size,weights='./trained_model/resnet_attention_56/resnet_attention_56.49-0.8762.hdf5')
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    # print(model.summary())
    result=format_results(generator,model)

    export_to_json(json_name,result)

with open(json_name,'r') as js:
    text = json.loads(js.readline())
for file in text:
    list_dict[file['image_id']].append(file['disease_class'])

print("merge results")
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
json_name='combine_submit.json'
with open(json_name, 'w') as f:
    json.dump(result, f)
    print('write %s, num is %d' % (json_name,len(result)))

