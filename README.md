# keras-aichallenger-2018-plant-recognition

# environments

```
tensorflow-gpu                     1.9.0 
jupyter                            1.0.0 
Keras                              2.1.0
h5py                               2.7.1
ubuntu 16.04
gtx 1080ti
```
# tutorial

## datasets

1. at first, you need to download aichallenger plant datasets from the official site:

https://challenger.ai/competition/pdr2018

2. for dataset generation, please refer to data_split.ipynb

it's easy to modify for your purpose


## training

I provide various kinds of models for training, if you want to use one of these models, please refer to the train.py, I provide the example code here
```
python train.py --model_name=resnet34
```
## inference
before you use the following command, you should manually add one line code to load your trained model, for example:

model=load_model('./trained_model/resnet34/resnet34.54-0.8773.hdf5')

```
python inference.py
```
## results combinations
please refer to the 
> create_csv_results.py
> csv_combinations.py

## grade 
- densenet161+inception_resnet_v2+seresnet50+resnet50+resnet34: 0.0.8817

- 我这里就当做baseline开源了
