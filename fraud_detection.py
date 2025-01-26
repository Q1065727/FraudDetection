import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE

results = []

# Function to append results after training each model
def append_results(model_name, y_test, y_pred, y_prob, metrics):
    result = {
        'Model': model_name,
        'Accuracy': metrics[0],
        'Precision': metrics[1],
        'Recall': metrics[2],
        'F1-Score': metrics[3],
        'ROC-AUC': metrics[4],
        'Confusion Matrix': metrics[5].tolist(),  # Convert matrix to list for storage
        'Predictions': y_pred.tolist(),
        'Probabilities': y_prob.tolist(),
    }
    results.append(result)

# Function to load and preprocess data
def load_and_preprocess_data(file_path, sample_size=None):
    df = pd.read_csv(file_path) # Read the csv file from the given path
    if sample_size: # If sample is given as parameter then select random N number of samples from the dataset.
        df = df.sample(n=sample_size, random_state=42)

    # Change type column values into categorical data and set integers number in order to process it for machine learning.
    df['type'] = df['type'].astype('category').cat.codes

    # Remove isFraud, nameOrig and nameDest columns. (Receipent information is not needed for our training)
    X = df.drop(columns=['isFraud', 'nameOrig', 'nameDest'])
    y = df['isFraud']

    return X, y

# Splitting the data. %30 for test and %70 for training. It will split as the same every run with random state number.
def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Class imbalance method for improve performance (Generates synthetic samples for minority class)
def resample_data(X_train, y_train, sampling_strategy=0.3):
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    return smote.fit_resample(X_train, y_train)

# Calculate evaluation metrics by sklearn library.
def evaluate_model(y_test, y_pred, y_prob):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, roc_auc, conf_matrix

# Threshold adjustment
# Find the optimal threshold number and according to it play around with threshold in order to get desired predictions.
def adjust_threshold_and_evaluate(y_test, y_prob, optimal_threshold=0.95):
    y_pred_adjusted = (y_prob >= optimal_threshold).astype(int) # Compare y_prob with optimal threshold. Then convert booleans to 0 or 1.

    accuracy_adj, precision_adj, recall_adj, f1_adj, roc_auc_adj, conf_matrix_adj = evaluate_model(y_test, y_pred_adjusted, y_prob)

    return accuracy_adj, precision_adj, recall_adj, f1_adj, roc_auc_adj, conf_matrix_adj

# Gradient Boosting Model
def gradient_boosting_model(file_path, sample_size=None):
    print("Training Gradient Boosting Classifier...")

    # Call load and preprocess method to split the data.
    X, y = load_and_preprocess_data(file_path, sample_size)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Call SMOTE method to improve performance
    X_train_resampled, y_train_resampled = resample_data(X_train, y_train)

    # Training Gradient Boosting model. Number of trees: 100
    gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3, verbose=1)
    gb_model.fit(X_train_resampled, y_train_resampled)

    # Make the predictions by using X_test according to trained model
    y_pred = gb_model.predict(X_test)
    y_prob = gb_model.predict_proba(X_test)[:, 1] # Probabilities for positive classes.

    metrics = evaluate_model(y_test, y_pred, y_prob)
    append_results('Gradient Boosting', y_test, y_pred, y_prob, metrics)

    # Evaluate metrics for Gradient Boosting model
    accuracy, precision, recall, f1, roc_auc, conf_matrix = evaluate_model(y_test, y_pred, y_prob)
    print("\nEvaluation Metrics for Gradient Boosting Model (Before Adjustments):")
    print(f"Accuracy: {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1-Score: {f1:.5f}")
    print(f"ROC-AUC: {roc_auc:.5f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Adjust threshold
    optimal_threshold = 0.95  # Optimal (for me) threshold is assigned
    accuracy_adj, precision_adj, recall_adj, f1_adj, roc_auc_adj, conf_matrix_adj = adjust_threshold_and_evaluate(y_test, y_prob, optimal_threshold)

    # Evaluate metrics for Gradient Boosting model with adjusted threshold
    print("\nEvaluation Metrics After Threshold Adjustment for Gradient Boosting Model:")
    print(f"Accuracy: {accuracy_adj:.5f}")
    print(f"Precision: {precision_adj:.5f}")
    print(f"Recall: {recall_adj:.5f}")
    print(f"F1-Score: {f1_adj:.5f}")
    print(f"ROC-AUC: {roc_auc_adj:.5f}")
    print("Confusion Matrix After Threshold Adjustment:")
    print(conf_matrix_adj)

# Random Forest Model
def random_forest_model(file_path, sample_size=None):
    print("Training Random Forest Classifier...")

    # Call load and preprocess method to split the data.
    X, y = load_and_preprocess_data(file_path, sample_size)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Call SMOTE method to improve performance
    X_resampled, y_resampled = resample_data(X_train, y_train)

    # Training Random Forest model. Number of trees: 300 This is relatively faster to calculate when comparing to
    # Gradient Boost model therefore n_estimators is set to 300.
    rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced", verbose=1)
    rf_model.fit(X_resampled, y_resampled)

    # Make the predictions by using X_test according to trained model
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1] # Probabilities for positive classes

    metrics = evaluate_model(y_test, y_pred, y_prob)
    append_results('Random Forest', y_test, y_pred, y_prob, metrics)

    # Evaluate metrics for Random Forest model
    accuracy, precision, recall, f1, roc_auc, conf_matrix = evaluate_model(y_test, y_pred, y_prob)
    print("\nEvaluation Metrics for Random Forest Model (Before Adjustments):")
    print(f"Accuracy: {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1-Score: {f1:.5f}")
    print(f"ROC-AUC: {roc_auc:.5f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Adjust threshold
    optimal_threshold = 0.8
    accuracy_adj, precision_adj, recall_adj, f1_adj, roc_auc_adj, conf_matrix_adj = adjust_threshold_and_evaluate(y_test, y_prob, optimal_threshold)

    print("\nEvaluation Metrics After Threshold Adjustment for Random Forest Model:")
    print(f"Accuracy: {accuracy_adj:.5f}")
    print(f"Precision: {precision_adj:.5f}")
    print(f"Recall: {recall_adj:.5f}")
    print(f"F1-Score: {f1_adj:.5f}")
    print(f"ROC-AUC: {roc_auc_adj:.5f}")
    print("Confusion Matrix After Threshold Adjustment:")
    print(conf_matrix_adj)



    '''
    Output:
    
    Gradient Boost Algorithm Evaluation Metrics:
    
    Evaluation Metrics:
    Accuracy: 0.98904
    Precision: 0.10379
    Recall: 0.99384
    F1-Score: 0.18795
    Confusion Matrix:
    [[1885455   20896]
     [     15    2420]]
    Optimal Threshold: 0.98523
    Optimal Threshold: 0.95000
    
    Evaluation Metrics After Threshold Adjustment:
    Accuracy: 0.99918
    Precision: 0.62376
    Recall: 0.90144
    F1-Score: 0.73732
    Confusion Matrix After Threshold Adjustment:
    [[1905027    1324]
     [    240    2195]]
    
    Process finished with exit code 0
    '''







    '''
    Evaluation Metrics for Random Forest:
Accuracy: 0.99920
Precision: 0.62886
Recall: 0.90707
F1-Score: 0.74277
ROC-AUC: 0.99767

Confusion Matrix:
[[598827    409]
 [    71    693]]

Optimal Threshold: 0.1
Evaluation Metrics:
Accuracy: 0.99564
Precision: 0.22427
Recall: 0.98691
F1-Score: 0.36549
ROC-AUC: 0.99767

Confusion Matrix:
[[596628   2608]
 [    10    754]]
    '''


file_path = '/Users/safaorhan/Downloads/ML_Assignment/fraud_detection.csv'  # Path to the dataset location
gradient_boosting_model(file_path,1000000)

'''random_forest_model(file_path,1000000)
'''
results_df = pd.DataFrame(results)
'''
results_df.to_csv('model_results2.csv', index=False)
print("Results saved to 'model_results2.csv'")'''

