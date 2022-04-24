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
<p>A feedforward neura network is built and modified for regression problem. The model is trained with batch size of 64 and epochs for 50 with application of early stopping to prevent from overfitting. 
</p>

## 4. Results
<p>The evaluation result obtained is shown in figure below.</p>
 
