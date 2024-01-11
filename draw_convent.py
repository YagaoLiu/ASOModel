from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

plotdata=np.load("curvedata.py.npz")
fpr = plotdata["fpr"]
tpr = plotdata["tpr"]
recall = plotdata["recall"]
precision = plotdata["precision"]

roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 16})
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'r',  label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'--', color = 'black')
plt.xlim([0, 1])
plt.ylim([0, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("C:/Users/yagao/Documents/ASO/AUC_ROC_CNN.png")


auc_pr = auc(recall, precision)
# Create the plot
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 16})
plt.plot(recall, precision,'r', label=f'PR Curve (AUC = {auc_pr:.2f})')
plt.title('Precision-Recall Curve')
plt.plot([0, 1], [0.5, 0.5],'--', color = 'black')
plt.xlim([0, 1])
plt.ylim([0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'lower right')
plt.savefig("C:/Users/yagao/Documents/ASO/AUC_PR_CNN.png")
