import pandas as pd
import numpy as np
import json

file_dir='results/densenet161_submit.csv'
df=pd.read_csv(file_dir)
densenet161_data={}
for indexs in df.index:
    filename=df.loc[indexs].values[0]
    predictions=df.loc[indexs].values[1:]
    densenet161_data[filename]=predictions


file_dir='results/inception_resnet_v2_submit.csv'
df=pd.read_csv(file_dir)
inception_v3_data={}
for indexs in df.index:
    filename=df.loc[indexs].values[0]
    predictions=df.loc[indexs].values[1:]
    inception_v3_data[filename]=predictions

file_dir='results/seresnet50_submit.csv'
df=pd.read_csv(file_dir)
seresnet50_data={}
for indexs in df.index:
    filename=df.loc[indexs].values[0]
    predictions=df.loc[indexs].values[1:]
    seresnet50_data[filename]=predictions


file_dir='results/resnet50_submit.csv'
df=pd.read_csv(file_dir)
resnet50_data={}
for indexs in df.index:
    filename=df.loc[indexs].values[0]
    predictions=df.loc[indexs].values[1:]
    resnet50_data[filename]=predictions

file_dir='results/resnet34_submit.csv'
df=pd.read_csv(file_dir)
resnet34_data={}
for indexs in df.index:
    filename=df.loc[indexs].values[0]
    predictions=df.loc[indexs].values[1:]
    resnet34_data[filename]=predictions

# file_dir='results/xception_submit.csv'
# df=pd.read_csv(file_dir)
# xception_data={}
# for indexs in df.index:
#     filename=df.loc[indexs].values[0]
#     predictions=df.loc[indexs].values[1:]
#     xception_data[filename]=predictions

result=[]
for k,v in seresnet50_data.items():
#     print(k)
#     print(v)
    temp_dict = {}
    x1=densenet161_data[k]
    x2=inception_v3_data[k]
    x3=seresnet50_data[k]
    x4=resnet34_data[k]
    # x5=resnet50_data[k]
    # x6=xception_data[k]

    x8=np.add(x1,x2)
    x9=np.add(x3,x4)
    
    # x11=np.add(x8,x9)
    merge=np.add(x8,x9)
    
    merged_sorted=np.argmax(merge)
    
    temp_dict['image_id'] = k.split('/')[-1]
    temp_dict['disease_class'] = int(merged_sorted)
    result.append(temp_dict)


json_name='submit.json'
with open(json_name, 'w') as f:
    json.dump(result, f)
    print('write %s, num is %d' % (json_name,len(result)))