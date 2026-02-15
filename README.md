Bank Marketing Classification – Machine Learning Assignment

1️⃣ Problem Statement



The objective of this project is to build and compare multiple machine learning classification models to predict whether a bank customer will subscribe to a term deposit based on marketing campaign data.



This is a binary classification problem where:



Target Variable (y):



yes → Customer subscribed



no → Customer did not subscribe



The project involves:



Data exploration and cleaning



Implementation of six machine learning models



Performance evaluation using multiple metrics



Deployment using Streamlit Community Cloud



2️⃣ Dataset Description



Dataset Name: Bank Marketing Dataset

Source: UCI Machine Learning Repository



Dataset Characteristics:



Number of Instances: 41,188



Number of Features: 20 input features



Target Variable: y (Yes/No)



Classification Type: Binary Classification



Feature Categories:



Client Information:



Age



Job



Marital status



Education



Default



Housing loan



Personal loan



Last Contact Information:



Contact type



Month



Day of week



Duration



Campaign Information:



Campaign



Pdays



Previous



Poutcome



Socioeconomic Attributes:



Employment variation rate



Consumer price index



Consumer confidence index



Euribor 3 month rate



Number of employees



The dataset contains categorical and numerical features.

Categorical variables were encoded using One-Hot Encoding.



3️⃣ Machine Learning Models Implemented



The following six classification models were implemented on the same dataset:



Logistic Regression



Decision Tree Classifier



K-Nearest Neighbors (KNN)



Naive Bayes (GaussianNB)



Random Forest (Ensemble Model)



XGBoost (Ensemble Boosting Model)



 4️⃣ Model Performance Comparison



| ML Model Name       | Accuracy | AUC    | Precision | Recall  | F1 Score | MCC    |
|----------------------|----------|--------|-----------|---------|----------|--------|
| Logistic Regression  | 0.9158   | 0.9423 | 0.7053    | 0.4332  | 0.5367   | 0.5110 |
| Decision Tree        | 0.8940   | 0.7427 | 0.5286    | 0.5474  | 0.5379   | 0.4781 |
| KNN                  | 0.9009   | 0.8346 | 0.6053    | 0.3470  | 0.4411   | 0.4094 |
| Naive Bayes          | 0.7476   | 0.8547 | 0.2866    | 0.8330  | 0.4265   | 0.3839 |
| Random Forest        | 0.9127   | 0.9469 | 0.7802    | 0.3136  | 0.4473   | 0.4598 |
| XGBoost              | 0.9177   | 0.9487 | 0.6555    | 0.5679  | 0.6085   | 0.5646 |




5️⃣ Observations on Model Performance



| ML Model | Observation about Model Performance |
|-----------|--------------------------------------|
| Logistic Regression | Logistic Regression delivered strong and stable performance with high AUC (0.9423) and good precision (0.7053). However, recall (0.4332) is moderate, indicating conservative positive predictions. |
| Decision Tree | Decision Tree achieved balanced precision and recall (~0.53–0.55). However, its lower AUC (0.7427) suggests weaker generalization compared to ensemble models. |
| KNN | KNN showed reasonable accuracy but relatively low recall (0.3470), which reduced its F1 Score. High-dimensional one-hot encoded features likely affected its distance-based calculations. |
| Naive Bayes | Naive Bayes achieved very high recall (0.8330) but extremely low precision (0.2866), indicating many false positives. This imbalance reduced overall accuracy and MCC. |
| Random Forest | Random Forest achieved the highest precision (0.7802) but very low recall (0.3136), meaning it predicted fewer positive cases. While AUC is high (0.9469), its F1 score is limited due to recall imbalance. |
| XGBoost | XGBoost achieved the best overall performance with the highest AUC (0.9487), highest F1 Score (0.6085), and highest MCC (0.5646). It provides the best balance between precision and recall among all models. |






6️⃣ Conclusion





Overall, ensemble methods outperformed individual classifiers. Although Random Forest achieved very high precision, its low recall reduced its overall effectiveness. XGBoost emerged as the most reliable model due to its superior balance across AUC, F1 Score, and MCC, making it the most suitable model for this classification task.




