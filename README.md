# face-recognition
# Galib, Marnim
# 1000-427-030
# 2017-11-27
# Assignment_06_03


Step 1 : Run this to reduce the size of  the training images of the CSE 5368 class to 160X160

python src/align/align_dataset_mtcnn.py datasets/class_train datasets/class_train_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25

Reduced images are saved in the directory : Galib_06/datasets/class_train_160

Step 2 : Using the LFW model to generate features for the CSE 5368 students images : the trained classification model is saved as a python pickle (Galib_06_01.pkl)


python src/classifier.py TRAIN datasets/class_train_160 models/facenet/20170512-110547/20170512-110547.pb models/Galib_06_01.pkl --batch_size 1000

Python pkl is saved in the directory : Galib_06/models

python src/classifier.py CLASSIFY datasets/class_train_160 models/facenet/20170512-110547/20170512-110547.pb models/marnim_classifier.pkl --batch_size 1000

Step 3 : Take Test image from webcam using the following command :

python src/Galib_06_01.py

Step 4 : Align the original test images in a 160X160 image

python src/align/align_dataset_mtcnn160.py datasets/test_images datasets/test_resize --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25


Step 5 : Draw a 160X160 box around the face of the original test images

python src/align/align_dataset_mtcnn.py datasets/test_images datasets/test_images_boxed --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25

Step 6 : Finally, show the test images side by side with the matching face from the training set.

python src/Galib_06_02.py CLASSIFY datasets/test_resize models/facenet/20170512-110547/20170512-110547.pb models/Galib_06_02.pkl --batch_size 1000
![image](https://user-images.githubusercontent.com/5978690/162720450-3cdcda12-e83c-4bed-875f-30ba7bb70459.png)
