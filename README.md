# Credit Card Fraud Detection
This project aims to detect fraudulent credit card transactions using machine learning models. The project involves data preprocessing, balancing class distributions, anomaly detection, and training multiple classifiers with hyperparameter tuning. The best model is saved for deployment.

## Dataset
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data). It contains transactions made by credit cards in September 2013 by European cardholders. 

* It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. 
* The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. 
* It contains only numerical input variables which are the result of a PCA transformation. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. 
* Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.


## Usage
1. Clone the repository
```
git clone https://github.com/yourusername/credit_card_fraud_detection.git
cd credit_card_fraud_detection
```

2. Create a virtual environment and install dependencies
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

3. Download the dataset 
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) and place the 'creditcard.csv' in data/ directory.

## Organization of Directory


## References

[Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/Introduction.html)
