# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import pickle
    
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report

# data = pd.read_csv('gesture_data.csv')

# d = data.iloc[:, :-1]
# labels = data.iloc[:, -1]

# x_train, x_test, y_train, y_test = train_test_split(d, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('Accuracy score: ', score * 100)

# with open('gesture_model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# # 1. Get predictions from your TEST set (not the training set!)
# # Assuming X_test and y_test were created during your training phase
# y_pred = model.predict(x_test)

# # 2. Create the matrix
# cm = confusion_matrix(y_test, y_pred)

# # 3. Plot it using Seaborn for a nice heatmap
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=model.classes_, 
#             yticklabels=model.classes_)

# plt.xlabel('Predicted Gestures')
# plt.ylabel('Actual Gestures')
# plt.title('Gesture Recognition Confusion Matrix')
# plt.show()

# # 4. Print the text report for precision/recall
# print(classification_report(y_test, y_pred))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 1 hand model

print("=" * 50)
print("TRAINING SINGLE-HAND MODEL")
print("=" * 50)

single_data = pd.read_csv('backend/gesture_data_single.csv')

# Drop metadata columns, keep features + label
X_single = single_data.drop(['seq_id', 'frame_idx', 'label'], axis=1)
y_single = single_data['label']

print(f"Single-hand data shape: {X_single.shape}")
print(f"Classes: {y_single.unique()}")
print(f"Class distribution:\n{y_single.value_counts()}")

# Split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_single, y_single, test_size=0.2, shuffle=True, stratify=y_single, random_state=42
)

# Train
model_single = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model_single.fit(X_train_s, y_train_s)

# Evaluate
y_pred_s = model_single.predict(X_test_s)
score_s = accuracy_score(y_test_s, y_pred_s)
print(f'\nSingle-hand Accuracy: {score_s * 100:.2f}%')

# Save
with open('gesture_model_single.pkl', 'wb') as f:
    pickle.dump(model_single, f)

# Confusion matrix
cm_s = confusion_matrix(y_test_s, y_pred_s)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_s, annot=True, fmt='d', cmap='Blues',
            xticklabels=model_single.classes_,
            yticklabels=model_single.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Single-Hand Gesture Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_single.png')
plt.show()

print(classification_report(y_test_s, y_pred_s))


# 2 hand model

print("\n" + "=" * 50)
print("TRAINING DUAL-HAND MODEL (One-Class)")
print("=" * 50)

from sklearn.svm import OneClassSVM

dual_data = pd.read_csv('backend/gesture_data_dual.csv')

X_dual = dual_data.drop(['seq_id', 'frame_idx', 'label'], axis=1)

print(f"Dual-hand data shape: {X_dual.shape}")
print(f"Training One-Class SVM on: {dual_data['label'].unique()[0]}")

# One-Class SVM: learns boundary of "normal" (Hollow Purple Prepare)
# nu = expected outlier ratio (10% of data might be noisy)
model_dual = OneClassSVM(gamma='scale', nu=0.1)
model_dual.fit(X_dual)

# Save
with open('gesture_model_dual.pkl', 'wb') as f:
    pickle.dump(model_dual, f)

print(f"Dual-hand model saved: gesture_model_dual.pkl")
print(f"Features expected: 85")
print(f"Prediction: +1 = Hollow Purple Prepare, -1 = not merge")

# Optional: visualize decision scores
scores = model_dual.decision_function(X_dual)
print(f"Training samples decision score range: {scores.min():.2f} to {scores.max():.2f}")


# summary

print("\n" + "=" * 50)
print("TRAINING SUMMARY")
print("=" * 50)
print(f"Single-hand model: {len(model_single.classes_)} classes, {score_s*100:.1f}% accuracy")
print(f"  Model saved: gesture_model_single.pkl")
print(f"  Features expected: 42")
print(f"Dual-hand model: One-Class SVM (Hollow Purple Prepare detection)")
print(f"  Model saved: gesture_model_dual.pkl")
print(f"  Features expected: 85")
print(f"  Prediction: +1 = merge detected, -1 = not merge")