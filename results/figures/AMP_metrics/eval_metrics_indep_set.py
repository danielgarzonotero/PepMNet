#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import matthews_corrcoef

# Ruta del archivo (puede ser CSV o Excel)
file_path = 'ensemble_testing_prediction' #name of the file without extension

try:
    df = pd.read_csv(f"{file_path}.csv")
except Exception as e:
    print(f"Error loading CSV: {e}")
    print("Trying to load as Excel file instead...")
    
    # Si falla, intenta cargar el archivo como Excel
    try:
        df = pd.read_excel(f"{file_path}.xlsx")
    except Exception as e:
        print(f"Error loading Excel: {e}")
        raise ValueError("Both CSV and Excel file loading failed.")


# Get the true labels and prediction scores
y_true = df['Target'].values
y_scores = df['Average score'].values

# Round the predictions to 0 or 1
# Define the threshold
threshold = 0.5
# Convert scores to binary predictions using the threshold
y_pred = (y_scores >= threshold).astype(int)

# Calculate and display all metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

auc_roc = roc_auc_score(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)
conf_mat = confusion_matrix(y_true, y_pred)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)

# Calculate the AUC using roc_auc_score
roc_auc = roc_auc_score(y_true, y_scores)

# Calculate the AUC using auc
roc_auc_from_auc = auc(fpr, tpr)

# Calculate specificity
tn, fp, fn, tp = conf_mat.ravel()
specificity = tn / (tn + fp)

# Calculate MCC
mcc = matthews_corrcoef(y_true, y_pred)

print('\n///////// Evaluation Metrics ////////')
print(f"-AUC using roc_auc_score: {roc_auc:.4f}")
print(f"-AUC using auc: {roc_auc_from_auc:.4f}")
print(f"-Accuracy: {acc:.4f}")
print(f"-Precision: {prec:.4f}")
print(f"-Recall: {rec:.4f}")
print(f"-F1-score: {f1:.4f}")
print(f"-Area under the ROC curve (AUC-ROC): {auc_roc:.4f}")
print(f"-Average Precision: {ap:.4f}")
print(f"-Specificity: {specificity:.4f}")
print(f"-MCC: {mcc:.4f}")


# Plot the confusion matrix using matplotlib
plt.figure(figsize=(7, 7))
plt.imshow(conf_mat, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix', fontsize=20, pad=20)
plt.colorbar()

# Add labels to the axes
plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'], fontsize=16)
plt.yticks([0, 1], ['Target 0', 'Target 1'], fontsize=16)

# Add the values in the cells
for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_mat[i, j], horizontalalignment='center', color='black', fontsize=24)

plt.xlabel('Predicted', fontsize=26, labelpad=15)
plt.ylabel('Target', fontsize=26, labelpad=15)
plt.grid(False)
plt.show()

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)

# Plot the ROC curve
plt.figure(figsize=(7, 7))  # Increase figure size
plt.plot(fpr, tpr, color='green', lw=2, label=f'AUC-ROC Testing = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate', fontsize=26, labelpad=15)  # Increase x-axis label size
plt.ylabel('True Positive Rate', fontsize=26, labelpad=15)  # Increase y-axis label size
plt.title('Receiver Operating Characteristic (ROC)', fontsize=20, pad=30)  # Increase title size
plt.xticks(fontsize=22)  # Increase x-axis tick size
plt.yticks(fontsize=22)  # Increase y-axis tick size
plt.legend(loc="lower right", fontsize=20)  # Increase legend size
plt.show()

# %%
