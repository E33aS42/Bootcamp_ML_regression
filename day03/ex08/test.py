import numpy as np
from other_metrics import accuracy_score_, precision_score_, recall_score_, f1_score_
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(" *** Example 1 ***\n")
y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

# Accuracy
# your implementation
print("accuracy_score: ", accuracy_score_(y, y_hat))
## Output: 0.5
# sklearn implementation
print("sklearn: ", accuracy_score(y, y_hat))
## Output: 0.5

# Precision
# your implementation
print("Precision: ", precision_score_(y, y_hat))
## Output: 0.4
# sklearn implementation
print("sklearn: ", precision_score(y, y_hat))
## Output: 0.4

# Recall
# your implementation
print("Recall: ", recall_score_(y, y_hat))
# Output:
0.6666666666666666
# sklearn implementation
print("sklearn: ", recall_score(y, y_hat))
## Output: 0.6666666666666666

# F1-score
# your implementation
print("F1-score: ", f1_score_(y, y_hat))
## Output: 0.5

# sklearn implementation
print("sklearn: ", f1_score(y, y_hat))
## Output: 0.5


print("\n *** Example 2 ***\n")
y_hat = np.array(['norminet', 'dog', 'norminet',
                 'norminet', 'dog', 'dog', 'dog', 'dog'])
y = np.array(['dog', 'dog', 'norminet', 'norminet',
             'dog', 'norminet', 'dog', 'norminet'])

# Accuracy
# your implementation
print("accuracy_score: ", accuracy_score_(y, y_hat))
## Output: 0.625
# sklearn implementation
print("sklearn: ", accuracy_score(y, y_hat))
## Output: 0.625

# Precision
# your implementation
print("Precision: ", precision_score_(y, y_hat, pos_label='dog'))
## Output: 0.6
# sklearn implementation
print("sklearn: ", precision_score(y, y_hat, pos_label='dog'))
## Output: 0.6

# Recall
# your implementation
print("Recall: ", recall_score_(y, y_hat, pos_label='dog'))
## Output: 0.75
# sklearn implementation
print("sklearn: ", recall_score(y, y_hat, pos_label='dog'))
## Output: 0.75

# F1-score
# your implementation
print("F1-score: ", f1_score_(y, y_hat, pos_label='dog'))
## Output: 0.6666666666666665
# sklearn implementation
print("sklearn: ", f1_score(y, y_hat, pos_label='dog'))
## Output: 0.6666666666666665


print("\n *** Example 3 ***\n")
y_hat = np.array(['norminet', 'dog', 'norminet',
                 'norminet', 'dog', 'dog', 'dog', 'dog'])
y = np.array(['dog', 'dog', 'norminet', 'norminet',
             'dog', 'norminet', 'dog', 'norminet'])

# Accuracy
# your implementation
print("accuracy_score: ", accuracy_score_(y, y_hat))
# sklearn implementation
print("sklearn: ", accuracy_score(y, y_hat))

# Precision
# your implementation
print("Precision: ", precision_score_(y, y_hat, pos_label='norminet'))
## Output: 0.6666666666666666
# sklearn implementation
print("sklearn: ", precision_score(y, y_hat, pos_label='norminet'))
## Output: 0.6666666666666666

# Recall
# your implementation
print("Recall: ", recall_score_(y, y_hat, pos_label='norminet'))
## Output: 0.5
# sklearn implementation
print("sklearn: ", recall_score(y, y_hat, pos_label='norminet'))
## Output: 0.5

# F1-score
# your implementation
print("F1-score: ", f1_score_(y, y_hat, pos_label='norminet'))
## Output: 0.5714285714285715
# sklearn implementation
print("sklearn: ", f1_score(y, y_hat, pos_label='norminet'))
## Output: 0.5714285714285715
