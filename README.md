# ESINN : Exploring Subitizing in Neural Networks

## Lines
We have three branches:
1. Main
2. Regression - Solving the problem using regression
3. classification - Using classification

## Implementing Next Steps

Link to the plan : https://docs.google.com/document/d/1V-lSgaCfvsu_onxFxLnleFd1AfKDmV9DdGdjtpXIzTg/edit#

## Data Created

Please find the Data in the folders:
 * [tree-md]
 * [Data folder]
   * [.txt] X 1 file - Contains the specifications of images in the folder
   * [.csv] X 1 file - Contains image path and number of shapes
   * [.png] X 50,000 file - Images

## Relevant Paper Links

https://ieeexplore.ieee.org/abstract/document/9425331?casa_token=epVf0fUHo48AAAAA:KeDX4l_IXJx3krlXrc_VKvLSskenBnu8tDE8dsqDJ_-Goldh2zoUnXe0IDSyz_GWjWdzTzvA8g

https://ojs.aaai.org/index.php/AAAI/article/view/3928

## Log runs

![Image](https://github.com/varunuppala/ESINN/blob/main/image/W%26B%20Chart%2011_21_2022%2C%205_08_36%20PM.png)
Above is a visualization of the results of a hyperparameter search on the regression model for various batch sizes and learning rates. The models were trained on only 35 data points. This experiment was run to see if any configurations of the tested hyperparameters would allow the model to get zero loss on the training set. Based on these results, it appears that our model is too small for our task, because none of the tested models were able to achieve approximately zero loss on the tiny training set.

