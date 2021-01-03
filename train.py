# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys
sys.path.append('..')
from lenet import LeNet
from keras.callbacks import ModelCheckpoint


os.environ["CUDA_VISIBLE_DEVICES"]="0"


## 定义超参数
# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 20
INIT_LR = 1e-3
BS = 16
CLASS_NUM = 5
norm_size = 255

## 加载数据
def load_data(path):
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = int(imagePath.split(os.path.sep)[-2])       
        labels.append(label)
    
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)                         
    return data,labels


## 训练
def LeNet_train(aug,trainX,trainY,testX,testY):
    # initialize the model
    weights_path = './save_weights/leNet.ckpt'
    model = LeNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)

    print("[INFO] compiling model...")
    checkpoint_path = './save_weights/leNet.ckpt'
    # model.load_weights(checkpoint_path)
    # if os.path.exists(checkpoint_path):
    #     model.load_weights(checkpoint_path)
    #     # 若成功加载前面保存的参数，输出下列信息
    #     print("checkpoint_loaded")

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, # TODO:任务是一个多分类问题，可以使用类别交叉熵（categorical_crossentropy）。但如果执行的分类任务仅有两类，那损失函数应更换为二进制交叉熵损失函数（binary cross-entropy）
        metrics=["accuracy"])

    # 断点续训
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='acc', save_weights_only=True,verbose=1,save_best_only=True, period=1)

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY), validation_steps=12, steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1, callbacks=[checkpoint])

    # save the model to disk
    print("[INFO] serializing network...")
    # model.save("test_sign.model")
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

def ResNet_fineturning(aug,trainX,trainY,testX,testY):
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing import image
    from keras.models import Model
    from keras import Sequential
    from keras.layers import Dense, GlobalAveragePooling2D, Softmax, Dropout
    from keras import backend as K

    # 构建不带分类器的预训练模型
    base_model = ResNet50 (weights='imagenet', include_top=False)

    model = Sequential([base_model,
                        GlobalAveragePooling2D(),
                        Dropout(rate=0.5),
                        Dense(1024, activation="relu"),
                        Dropout(rate=0.5),
                        Dense(5),
                        Softmax()])

    # 我们只训练顶部的几层（随机初始化的层）
    # 锁住所有卷积层
    for layer in base_model.layers:
        layer.trainable = False

    # 编译模型（一定要在锁层以后操作）
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

    # 加载之前训练的权重
    checkpoint_path = './save_weights/ResNet.ckpt'
    # model.load_weights(checkpoint_path) 

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='acc', save_weights_only=True,verbose=1,save_best_only=True, period=1)

    # 在新的数据集上训练几代
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1, callbacks=[checkpoint])
    
    # save the model to disk
    print("[INFO] serializing network...")
    model.save_weights("./save_weights/ResNet.ckpt")
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

## main函数
#python train.py --dataset_train ../../traffic-sign/train --dataset_test ../../traffic-sign/test --model traffic_sign.model
if __name__=='__main__':
    train_file_path = "E:\\ML\\ML_design\\LeNet\\photos_data\\train"
    test_file_path = "E:\\ML\\ML_design\\LeNet\\photos_data\\val"
    print(os.listdir(train_file_path))
    trainX,trainY = load_data(train_file_path)
    testX,testY = load_data(test_file_path)
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    ResNet_fineturning(aug,trainX,trainY,testX,testY)