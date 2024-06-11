import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torchmetrics.classification import BinaryConfusionMatrix


def amp_evaluate_model(prediction, target, dataset_type, threshold, device):
    # Crear la matriz de confusión binaria
    bcm = BinaryConfusionMatrix(task="binary", threshold=threshold, num_classes=2).to(device) 
    confusion_matrix = bcm(prediction, target)
    confusion_matrix_np = confusion_matrix.detach().cpu().numpy()

    # Extraer TN, FP, FN, TP de la matriz de confusión
    TN = confusion_matrix[0, 0].cpu().numpy()
    FP = confusion_matrix[0, 1].cpu().numpy()
    FN = confusion_matrix[1, 0].cpu().numpy()
    TP = confusion_matrix[1, 1].cpu().numpy()

    # Añadir números a la matriz de confusión y guardarla
    cmap = plt.get_cmap('YlGn')
    plt.matshow(confusion_matrix_np, cmap=cmap)
    plt.title('Confusion Matrix Plot - {}'.format(dataset_type))
    plt.colorbar()
    for i in range(confusion_matrix_np.shape[0]):
        for j in range(confusion_matrix_np.shape[1]):
            plt.text(j, i, str(confusion_matrix_np[i, j]), ha='center', va='center', color='black', fontsize=18)
    plt.xlabel('Predicted Negative         Predicted Positive')
    plt.ylabel('True Positive               True Negative')
    plt.savefig('results/{}_bcm.png'.format(dataset_type), dpi=216)
    plt.show()
    

    # Cálculo de métricas de evaluación
    ACC = (TP + TN) / (TP + TN + FP + FN)
    PR = TP / (TP + FP)
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    F1 = 2 * (PR * SN) / (PR + SN)
    
    # Calcular Matthews Correlation Coefficient (MCC)
    mcc = (TP * TN - FP * FN) / \
                        ((TP + FP) * (TP + FN) * \
                         (TN + FP) * (TN + FN)) ** 0.5

    # Calcular la curva ROC
    fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), prediction.cpu().numpy())

    # Calcular el área bajo la curva ROC (AUC)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='mediumseagreen', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)',fontsize=14)
    plt.ylabel('True Positive Rate (TPR)',fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve - {} Set'.format(dataset_type))
    plt.legend(loc='lower right', fontsize=16)
    plt.savefig('results/{}_ROC.png'.format(dataset_type), dpi=216)
    plt.show()

    # Imprimir las métricas
    print('/// Evaluation Metrics - {} ///\n'.format(dataset_type)) 
    print(f"Accuracy: {ACC:.3f}")
    print(f"Precision: {PR:.3f}")
    print(f"Recall: {SN:.3f}")
    print(f"Specificity: {SP:.3f}")
    print(f"F1 Score: {F1:.3f}")
    print(f"MCC: {mcc:.3f}")
    
    return  TP, TN, FP, FN, ACC, PR, SN, SP,F1, mcc, roc_auc

