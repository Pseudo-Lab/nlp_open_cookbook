# Changelog

## Model Performance Comparison - 04.20.2022

- Write testcode to compare accuracies of different models
    - Used CountVectorizer to convert text to vector
    - Model return only "긍정": 0.5
    - Logistic Regression: 0.765
    - Decision Tree: 0.671
    - Random Forest: 0.698
- Possible reasons why LR performs better than DT or RF
    - Logistic regression is more robust to outliers
    - Tree model is easily affected by noise
    - Need to embedding or preprocess to remove outliers and make patterns(stopwords etc)

