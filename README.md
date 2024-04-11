# About the drop width estimation project

Capturing side-view videos from sliding drops offers a robust method to researchers for investigating the surface and drop dynamics. However, in some studies, it's essential to examine certain aspects like the width of the drop from a front-view perspective as well. The drop's width is a crucial parameter because of its association with the lateral adhesion force and the friction force.  In such situations, the common practice of incorporating extra cameras or mirrors to monitor changes in the drop from a front-view perspective is often cumbersome and reduces the viewing area. Also, this limitation can impede a comprehensive analysis of sliding drops, especially when dealing with scenarios that entail surface defects. 
We proposed an approach that eliminates the need to incorporate additional equipment into the experimental setup and, also, ensures the viewing area remains unrestricted. The Long Short Term Memory (LSTM) model with a 20-sliding window has an error of 67 µm based on RMSE. Within the spectrum of drop widths in our dataset, ranging from 1.6 mm to 4.4 mm, this RMSE indicates that with our approach we can predict the width of sliding drops with an error of 2.4%. Furthermore, the applied LSTM model provides a drop width across the whole sliding length of 5 cm, previously unattainable.


https://github.com/AK-Berger/Drop_width_estimation/assets/57271994/85117226-5a3e-46f6-be1c-7ed3f92c787b



---
# Publication Information:

Title: Estimating front-view drop width via side-view features using recurrent neural networks

Authors: Sajjad Shumaly, Fahimeh Darvish, Xiaomei Li, Oleksandra Kukharenko, Werner Steffen, Hans-Jürgen Butt, Rüdiger Berger*

Journal: ---

Publication Date: Jun --, 2024

DOI: ---------------------

---
# Data Information:

### The tutorial

- tutorial.ipynb

    The tutorial provides a detailed, step-by-step explanation of how we trained the LSTM model with a 20-slide window, which was determined to be the best model based on RMSE. Using this file, reviewers can access the code, variables, and hyperparameters for examination. Furthermore, the document demonstrates how we utilized the trained model to incorporate the final validation metrics and estimate drop width.

### The LSTM weights

- LSTM weights.h5

    The "LSTM weights.h5" file represents the fully trained 20-slide window LSTM model that can be employed by others for the purpose of estimating drop width.

### The dataset

- Dataset.xlsx

    The "Dataset.xlsx" file represents the dataset we compiled after processing and integrating the sliding drop videos. In this dataset, the "Status" column indicates whether a video is associated with training, testing, or final validation measurements. Initially, we made random selections for these assignments but later maintained consistency across all algorithms to ensure a fair comparison across different models. It's worth noting that the final validation records differ from the regular validation records. After dividing the dataset into testing and training subsets, we further split the training data into the typical training and validation sets for the training process. Final validation involves measurements conducted externally to the dataset, serving to assess the model's validity.

  ---
# Dependencies 

- tensorflow 2.5.0; https://pypi.org/project/tensorflow/

- keras 2.9.0; https://pypi.org/project/keras/

- cv2 4.5.4; https://pypi.org/project/opencv-python/

- scipy 1.7.1; https://pypi.org/project/scipy/

- PIL 8.4.0; https://pypi.org/project/PIL/

- numpy 1.20.3; https://pypi.org/project/numpy/

- pandas 1.3.4; https://pypi.org/project/pandas/

- matplotlib 3.4.3; https://pypi.org/project/matplotlib/

---
# Support

You can communicate with us using the following e-mails:

- shumalys@mpip-mainz.mpg.de
- berger@mpip-mainz.mpg.de
---
