## Intel Image classification

![images](../images/image_classification.jpg)

For seeing code implementation visit my 
[kaggle notebook](https://www.kaggle.com/billiemage/pytorch-use-pretrained-model).

* For this project i used [Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification) dataset, which contains image data divided in 3 sub-folder `seg_pred`,`seg_test` and `seg_train`.

* This is 6 class classification project where i have 6 different classes of images that is   `mountain`, `street`,`buildings`, `sea`, `forest`, `glacier`.

* The given images weren't the same sizes so i resized them all by `150x150 px` and for increase the efficiency of model on not so clear images `RandomHorizontalFlip` has been used.

* train and validation data divided in 85:15 ratio.

* For optimization `stochastic gradient descent` has been used with  `MultiStepLR` scheduler with gamma = 0.06.


| *Data Link*  |  https://www.kaggle.com/puneet6060/intel-image-classification |
|---|---|
|  *Kaggle Notebook Link* |  https://www.kaggle.com/billiemage/pytorch-use-pretrained-model |
