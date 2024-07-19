import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score

# Seed for reproducibility
random.seed(42)

# Load your credit card fraud detection dataset from Kaggle
# Assuming the dataset is named 'credit_card_fraud_data.csv'

# Load data from CSV file
df = pd.read_csv('creditcard.csv')

# Feature selection
# Assuming 'Amount' and 'Class' are relevant features
features = ['Amount']  # Add more features as needed
X = df[features]
y = df['Class']

# Training-testing ratios
ratios = [0.4, 0.3, 0.2, 0.1]

results = {}

for ratio in ratios:

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reinforcement Learning (Q-learning)
    def q_learning_train(X, y, episodes=100, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        n_actions = 2  # Assuming binary classification (fraud or not fraud)
        n_states = X.shape[0]
        
        # Initialize Q-table with zeros
        Q = np.zeros((n_states, n_actions))
        
        for _ in range(episodes):
            state = random.randint(0, n_states - 1)
            while True:
                if random.uniform(0, 1) < epsilon: # epsilon is our exploration rate
                    action = random.randint(0, n_actions - 1)  # Explore
                else:
                    action = np.argmax(Q[state, :])  # Exploit
                
                reward = y.iloc[state] if action == 1 else -y.iloc[state]
                
                next_state = (state + 1) % n_states
                Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
                
                state = next_state
                if state == 0:
                    break
        
        return Q

    def q_learning_predict(Q, X):
        '''Predicts the actions for each state in X (test data) using the Q-table.'''
        y_pred = []
        for state in range(X.shape[0]):
            action = np.argmax(Q[state, :])
            y_pred.append(action)
        return np.array(y_pred)

    # Train Q-learning model
    Q = q_learning_train(pd.DataFrame(X_train), y_train)
    y_pred_q = q_learning_predict(Q, pd.DataFrame(X_test))

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred_q)
    print(f"Confusion Matrix for ratio {ratio}:\n", cm)

    # Calculate metrics
    precision_q = precision_score(y_test, y_pred_q)
    accuracy_q = accuracy_score(y_test, y_pred_q)
    recall_q = recall_score(y_test, y_pred_q)
    f1_q = f1_score(y_test, y_pred_q)
    
    results[ratio] = {'precision': precision_q, 'accuracy': accuracy_q, 'recall': recall_q, 'f1': f1_q}

# Print results
print("Results:")
print("Ratio Precision Accuracy Recall F1")
for ratio, metrics in results.items():
    print(f"{ratio:}\t{metrics['precision']:.4f}\t{metrics['accuracy']:.4f}\t{metrics['recall']:.4f}\t{metrics['f1']:.4f}")
print("Average F1", round(sum([results[ratio]['f1'] for ratio in results.keys()])/len(results.keys()), 4))
print("Average Accuracy", round(sum([results[ratio]['accuracy'] for ratio in results.keys()])/len(results.keys()), 4))
print("Average Precision", round(sum([results[ratio]['precision'] for ratio in results.keys()])/len(results.keys()), 4))
print("Average Recall", round(sum([results[ratio]['recall'] for ratio in results.keys()])/len(results.keys()), 4))