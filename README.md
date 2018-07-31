# ML
深度学习

转载请注明：https://blog.csdn.net/wyx100/article/details/80428424

效果展示



未完待续。。。
环境配置

win7sp1

python                 3.6.3
dlib                      19.7.0  
tensorflow            1.3.0rc0
keras                     2.1.5 
opencv-python      3.4.1+contrib
pillow                    4.2.1
numpy                   1.14.1+mkl
numpy                   1.12.1

软件下载地址：https://www.lfd.uci.edu/~gohlke/pythonlibs/

项目实现步骤

1.整理人脸图片（格式：jpg，150x150）

每个人物100张（80张训练（train），20张验证（validation））

完整项目下载

为方便没积分童鞋，请加企鹅，共享文件夹。

包括：代码、数据集合（图片）、已生成model、安装库文件等。

data 

      train

            fsm

                    0.jpg

                    1.jpg

                     。。。

                    79.jpg

            gje

            qyy

    validation

            fsm

                    80.jpg

                    81.jpg

                     。。。

                    99.jpg

            gje

            qyy

2.训练CNN（tensorflow、keras）模型

3.基于2训练的model（dlib检测人脸）识别视频某张图片中的人脸，并标记姓名

    1）打开视频，截取一帧图片

    2）检测1）中图片的人脸（1张或多张），未检测到人脸则结束本次循环

    3）基于2训练的model识别图片中的人脸，并标记姓名

    4）输出框出人脸并标记姓名的图片。

代码

1. model_cnn_train.py

    使用卷积神经网络训练人脸识别（不是检测）模型（模型结构见文章末尾）
    
根据keras2.1.6中example（点击下载）下cifar10_cnn.py修改

英文文档

https://keras.io/

https://pypi.org/project/Keras/

https://pypi.org/project/Keras/#files

https://github.com/keras-team/keras.git （代码下载）

中文文档

http://keras-cn.readthedocs.io/en/latest/

6个周期可以达到99%的准确率。

2. video_face_sign.py

使用dlib检测视频中的人脸，调用1中的训练的模型判断对应人（是谁）并标记中文姓名。

备注：通过模型（可以使用leNet、vgg16等网络）、样本质量、样本数量、样本多样性调整可优化实际识别效果。



