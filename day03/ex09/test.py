import numpy as np
from sklearn.metrics import confusion_matrix
from confusion_matrix import confusion_matrix_


y_hat = np.array([['norminet'], ['dog'], ['norminet'],
                 ['norminet'], ['dog'], ['bird']])
y = np.array([['dog'], ['dog'], ['norminet'], [
             'norminet'], ['dog'], ['norminet']])

print("\n *** Example 1 ***\n")
# your implementation
print("confusion matrix: \n", confusion_matrix_(y, y_hat))
# Output:
# array([[0 0 0]
# [0 2 1]
# [1 0 2]])
# sklearn implementation
# print("sklearn: \n", confusion_matrix(y, y_hat))
# Output:
# array([[0 0 0]
# [0 2 1]
# [1 0 2]])

print("\n *** Example 2 ***\n")
# your implementation
print("confusion matrix: \n", confusion_matrix_(
    y, y_hat, labels=['dog', 'norminet']))
# Output:
# array([[2 1]
# [0 2]])
# sklearn implementation
# print("sklearn: \n", confusion_matrix(y, y_hat, labels=['dog', 'norminet']))
# Output:
# array([[2 1]
# [0 2]])

print("\n *** Example 3 ***\n")
print("confusion matrix: \n", confusion_matrix_(y, y_hat, df_option=True))
# Output:
# 			bird 	dog 	norminet
# bird 		0 		0 		0
# dog 		0		2 		1
# norminet	1 		0 		2

print("\n *** Example 4 ***\n")
print("confusion matrix: \n", confusion_matrix_(
    y, y_hat, labels=['bird', 'dog'], df_option=True))
# Output:
# 		bird 	dog
# bird	0		0
# dog	0		2
