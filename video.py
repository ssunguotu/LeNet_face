#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from retinaface import Retinaface
from PIL import Image
from keras.models import load_model
from predict import norm_size
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import imutils
import time
import cv2
from lenet import LeNet
from train import CLASS_NUM
mp = {'[0]':'sun', '[1]':'wang', '[2]':'gong', '[3]':'xie', '[4]':'xu'}

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
retinaface = Retinaface()

MODEL = 'ResNet'
# model = load_model("test_sign.model") ## TODO: TEMP
if MODEL == 'LeNet':
    model = LeNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
else:
###
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing import image
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    from keras import backend as K

    # 构建不带分类器的预训练模型
    base_model = ResNet50 (weights='imagenet', include_top=False)

    # 添加全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 添加一个全连接层
    x = Dense(1024, activation='relu')(x)

    # 添加一个分类器，假设我们有200个类
    predictions = Dense(5, activation='softmax')(x)

    # 构建我们需要训练的完整模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 首先，我们只训练顶部的几层（随机初始化的层）
    # 锁住所有 InceptionV3 的卷积层
    for layer in base_model.layers:
        layer.trainable = False
###

print("[INFO] compiling model...")
checkpoint_path = './save_weights/leNet.ckpt'
model.load_weights(checkpoint_path)

# 调用摄像头
capture=cv2.VideoCapture(0) # capture=cv2.VideoCapture("1.mp4")

fps = 0.0
while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # 进行检测, frame用于显示先验框, image_clip用于分类
    frame, image_clip, pnum = retinaface.detect_image(frame)
    frame = np.array(frame)

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    print(pnum)
    if pnum > 0:
        ######################
        # pre-process the image for classification
        image_clip = cv2.resize(image_clip, (norm_size, norm_size))
        image_clip = image_clip.astype("float") / 255.0
        image_clip = img_to_array(image_clip)
        image_clip = np.expand_dims(image_clip, axis=0)

        # classify the input image
        result = model.predict(image_clip)[0]
        #print (result.shape)
        proba = np.max(result)
        label = str(np.where(result==proba)[0])
        print(label)
        label = "{}: {:.2f}%".format(mp[label], proba * 100)
        print(label)
        frame = cv2.putText(frame, label, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ######################

    cv2.imshow("video",frame)

    c= cv2.waitKey(1) & 0xff 
    if c==27:
        capture.release()
        break
