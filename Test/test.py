import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# your true and predicted labels
true_labels = [0,1,2,3,4,5,5,5]
predicted_labels = [0,1,2,3,4,5,5,4]

# your mapping
mapping = {'O': 0, 'rna': 1, 'dna': 2, 'cell_line': 3, 'cell_type': 4, 'protein': 5}

# create the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()

# reverse the mapping to get the labels from the values
reverse_mapping = {v: k for k, v in mapping.items()}
print(reverse_mapping)
# set the x and y axis labels using the reverse_mapping
ax = plt.gca()
print([reverse_mapping[i] for i in range(len(mapping))])
#set xticks and yticks according to the mapping
ax.set_xticks([i for i in range(len(mapping))])
ax.set_yticks([i for i in range(len(mapping))])
ax.set_xticklabels([reverse_mapping[i] for i in range(len(mapping))])
ax.set_yticklabels([reverse_mapping[i] for i in range(len(mapping))])


#ax.set_xticklabels([reverse_mapping[i] for i in range(len(mapping))])
#ax.set_yticklabels([reverse_mapping[i] for i in range(len(mapping))])
plt.show()
