import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Feature selection
X = df.drop(columns=['Class'])
y = df['Class']

# Training-testing ratios
ratios = [0.4, 0.3, 0.2, 0.1]

results = {}

for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=0)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model training
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for ratio {ratio}:\n", cm)

    # Calculate metrics
    precision_q = precision_score(y_test, y_pred)
    accuracy_q = accuracy_score(y_test, y_pred)
    recall_q = recall_score(y_test, y_pred)
    f1_q = f1_score(y_test, y_pred)
    
    results[ratio] = {'precision': precision_q, 'accuracy': accuracy_q, 'recall': recall_q, 'f1': f1_q}

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print results
print("Results:")
print("Ratio Precision Accuracy Recall F1")
for ratio, metrics in results.items():
    print(f"{ratio}\t{metrics['precision']:.4f}\t{metrics['accuracy']:.4f}\t{metrics['recall']:.4f}\t{metrics['f1']:.4f}")
print("Average F1", round(sum([results[ratio]['f1'] for ratio in results.keys()]) / len(results.keys()), 4))
print("Average Accuracy", round(sum([results[ratio]['accuracy'] for ratio in results.keys()]) / len(results.keys()), 4))
print("Average Precision", round(sum([results[ratio]['precision'] for ratio in results.keys()]) / len(results.keys()), 4))
print("Average Recall", round(sum([results[ratio]['recall'] for ratio in results.keys()]) / len(results.keys()), 4))

# Assuming final_results_table exists and is a DataFrame
# Uncomment the following lines if you have a DataFrame named final_results_table
# final_results_table.loc[len(final_results_table)] = ['Random Forest', 
#                                                      round(sum([results[ratio]['accuracy'] for ratio in results.keys()]) / len(results.keys()), 4),
#                                                      round(sum([results[ratio]['precision'] for ratio in results.keys()]) / len(results.keys()), 4),
#                                                      round(sum([results[ratio]['recall'] for ratio in results.keys()]) / len(results.keys()), 4),
#                                                      round(sum([results[ratio]['f1'] for ratio in results.keys()]) / len(results.keys()), 4)]
