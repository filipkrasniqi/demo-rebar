# Introduction
This repository shows how a RetinaNet can be built to solve a segmentation task on DemoRebar dataset.
The model has been implemented using Tensorflow + Keras frameworks.
Dependencies have been added to requirements.txt.

## Code
- [Dataset Initialization](https://github.com/filipkrasniqi/demo-rebar/blob/master/EDA/init-dataset.ipynb): allows to create the dataset useful for training, validation, test;
- [Training](https://github.com/filipkrasniqi/demo-rebar/blob/master/training/main.py): allows to train the model;
- [Inference script](https://github.com/filipkrasniqi/demo-rebar/blob/master/inference/main.py): allows to execute the pretrained model and output the bounding boxes and the object count;
- [Results](https://github.com/filipkrasniqi/demo-rebar/blob/master/EDA/results.ipynb): technical report showing model performances.
