# FIRE EYE 

https://github.com/Amerzish-25/Fire_Eye-A_Real-Time_Forest_Fire_and_Smoke_Detection_Website/assets/106583511/1827098d-b95a-447a-8d40-e50d3a37c4e7

## CONTENTS
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)

## INTRODUCTION
In this project, we propose a system for detecting and monitoring forest fires using two cutting-edge deep learning algorithms (YOLO-v5 and Inception-v3), designed to detect fires at their earliest stages. This early warning system will critically analyse the scope of artificial intelligence, reduce the response time of fire authorities, and help them mitigate the loss of life and property caused by forest fires before they spread uncontrollably. This system will increase the reliability and accuracy of fire detection and monitoring operations and contribute to the overall sustainability and resilience of forest ecosystems.
 
## FEATURES
- Includes real-time monitoring feature, using YOLO algorithm for early detection and rapid response to wildfires
- Include Location Tracker feature, in which GPS tracker is implemented to track the area and location of fire
- Include Fire Alerts Reports feature, which generate instant reporting for timely decision-making and effective fire record management
- Utilize the Inception v3 deep learning model in model prediction feature, to accurately detect and classify smoke, fire and non fire
- Improve the efficiency and effectiveness of forest management and firefighting operations

![Features](https://github.com/Amerzish-25/Fire_Eye-A_Real-Time_Forest_Fire_and_Smoke_Detection_Website/assets/106583511/f2a81553-b055-40a5-a900-8e91ed30d707)

## INSTALLATION 
1. Clone the repository:
```
git clone https://github.com/Amerzish-25/Fire_Eye-A_Real-Time_Forest_Fire_and_Smoke_Detection_Website.git
cd Fire_Eye-A_Real-Time_Forest_Fire_and_Smoke_Detection_Website
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
## USAGE 
1. To run Flask Server:
```
python App.py
```
2. Open your web browser and Navigate to `http://127.0.0.1:5000` to access the application.

## DATASET
For training the YOLO-v5 and Inception-v3 models, you can find the datasets on Kaggle:

- YOLO V5 Dataset: [Link to YOLO-v5 Dataset on Kaggle](https://www.kaggle.com/datasets/amerzishminha/fire-eye)
- Inception V3 Dataset: [Link to Inception-v3 Dataset on Kaggle](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset)

Please note that these datasets were used for training the respective models in this project. Ensure that you review the dataset licenses and terms on Kaggle before using them for your own purposes.

## MODEL TRAINING
### • YOLO-v5 Model Training
1. Download the YOLO training dataset from Kaggle
2. Set up **Yolov5_Training.ipynb** script for training the model from scratch. 
3. Use the following script to install the [YOLOv5](https://github.com/ultralytics/yolov5)
```
git clone https://github.com/ultralytics/yolov5  
cd yolov5
```
3. Install the required dependencies
```
pip install -r requirements.txt
```
4. Train the model
```
!python train.py --img 640 --batch 64 --epochs 70 --data custom_train.yaml --weights yolov5s.pt 
```
### • Inception-v3 Model Trraining
1. Download the Inception v3 training dataset from Kaggle
2. Set up **Inceptionv3_Model_Training.ipynb** script for training the model from scratch
3. Preprocess the Dataset using data augmentation
```
training_datagenarator= ImageDataGenerator(rescale=1./255,horizontal_flip=True, vertical_flip=True,shear_range=0.2,
                                           zoom_range=0.2,width_shift_range=0.2, height_shift_range=0.2,validation_split=0.1)
```
3. Load the Inception-v3 Model and configure model architecture
```
inceptionV3 = InceptionV3(input_shape=IMG_SIZE + [3], weights='imagenet', include_top=False)
```
4. Compile and Train the model
```
history =model.fit_generator(train, validation_data=validation, epochs=50, steps_per_epoch=train.samples//BATCH_SIZE, 
                    validation_steps=validation.samples//BATCH_SIZE, callbacks = callback)
```
5. Save the model for evaluation and prediction

## RESULTS
