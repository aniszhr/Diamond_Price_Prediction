# Diamond Price Prediction using Feedforward Neural Network

## 1. Summary
<p>The target of this project is to predict the price of diamond based on several features such as:- <br>
<ol>
<li>4C (carat, clarity, carat, color) </li>
<li>Dimensions (x,y,z) </li>
<li>Depth</li>
<li>Table</li>
</ol>

The dataset is obtained from [here](https://www.kaggle.com/datasets/shivam2503/diamonds).
</p>

## 2. IDE and Framework
<p>This project is created using Spyder as the main IDE. The frameworks mainly used in this project are Pandas, Numpy, Tensorflow Keras and Scikit-learn.</p>

## 3. Methodology

### 3.1 Data Pipeline
<p>The data is loaded and preprocessed to remove unwanted features. The categorical features are encoded ordinally. After that the data is then splitted into train-validation-test sets with a ratio of 60:20:20.</p>

### 3.2 Model Pipeline
<p>A feedforward neural network is built and modified for regression problem. The model is trained with batch size of 64 and epochs for 50 with application of early stopping to prevent from overfitting. The training stops at epoch 8 with training MAE of 794 and validation MAE of 524.
</p>

![model_p2](https://user-images.githubusercontent.com/72061179/165106242-a5e7df6b-8038-4c88-bb7b-49c09eea00d4.png)


## 4. Results
<p>The evaluation result obtained is shown in figure below.</p>
 
 ![img2](https://user-images.githubusercontent.com/72061179/164993514-5b0e9e74-f192-408d-b97f-80f4694c4e38.png)

 
 ![result](https://user-images.githubusercontent.com/72061179/164993508-aa3f80c1-bd91-4819-95aa-24276490a170.png)

