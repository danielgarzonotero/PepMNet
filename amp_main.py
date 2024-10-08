#%%
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.explain import CaptumExplainer, Explainer
from sklearn.model_selection import KFold
from torch.utils.data import  Subset
import numpy as np
import pandas as pd

from src.device import device_info
from src.data import GeoDataset_1
from src.amp_model import amp_pepmnet
from src.amp_process import amp_train, amp_validation, amp_predict_test
from src.amp_evaluation_metrics import amp_evaluate_model

# Get device information (GPU or CPU) and set it for model and data
device_information = device_info()
print(device_information)
device = device_information.device

start_time = time.time()

## SET UP DATALOADERS: 

# Load/Processing the dataset using GeoDataset_1, specifying the file paths for train and test datasets
# raw_name: Path to the CSV file containing the data
# index_x: The column index for the sequences in the CSV file
# index_y: The column index for the target property (AMP classification, 1 or 0)
# has_targets: Set to True because this is a supervised task with known targets (AMP classification)

datasets = {
            'test_ruiz': GeoDataset_1(raw_name='data/AMP/dataset Ruiz/Test_clean.csv',
                                            root='',
                                            index_x=0,
                                            index_y=1,
                                            has_targets= True
                                            ),
            'train_ruiz': GeoDataset_1(raw_name='data/AMP/dataset Ruiz/Train_clean.csv',
                                            root='',
                                            index_x=0,
                                            index_y=1,
                                            has_targets= True
                                            )
            }

# Assign the test dataset for later use in model evaluation
ruiz_testing_datataset = datasets['test_ruiz']

# Print the number of features for nodes and edges in the graph dataset (important for setting model dimensions)
print('Number of NODES features: ', ruiz_testing_datataset.num_features)
print('Number of EDGES features: ', ruiz_testing_datataset.num_edge_features)

# Calculate time spent on preprocessing the data
finish_time_preprocessing = time.time()
time_preprocessing = (finish_time_preprocessing - start_time) / 60 


# Set model input dimensions based on the number of node and edge features from the dataset
initial_dim_gcn = ruiz_testing_datataset.num_features
edge_dim_feature = ruiz_testing_datataset.num_edge_features

# Define hidden dimensions 
hidden_dim_nn_1 = 20
hidden_dim_nn_2 = 10

hidden_dim_gat_0 = 50

hidden_dim_fcn_1 = 200
hidden_dim_fcn_2 = 100
hidden_dim_fcn_3 = 10
dropout = 0

# Define a function to initialize the model with the specified architecture and optimizer
def initialize_model(
                    initial_dim_gcn,
                    edge_dim_feature,
                    hidden_dim_nn_1,
                    hidden_dim_nn_2,
                    hidden_dim_gat_0,
                    hidden_dim_fcn_1,
                    hidden_dim_fcn_2,
                    hidden_dim_fcn_3,
                    dropout,

                    learning_rate,
                    weight_decay,
                    seed,
                    ):
    
    torch.manual_seed(seed)

    model = amp_pepmnet(
                    initial_dim_gcn,
                    edge_dim_feature,
                    hidden_dim_nn_1,
                    hidden_dim_nn_2,
                    
                    hidden_dim_gat_0,
                    
                    hidden_dim_fcn_1,
                    hidden_dim_fcn_2,
                    hidden_dim_fcn_3,
                    dropout
                    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return model, optimizer



#/////////////////// Training /////////////////////////////
# Set up optimizer:
learning_rate = 1E-3 
weight_decay = 1E-5 
batch_size = 100

# Testing dataset setup:
ruiz_test_dataloader = DataLoader(ruiz_testing_datataset , batch_size, shuffle=False)

# Start timing for training process:
start_time_training = time.time()

# Training dataset setup:
ruiz_training_datataset = datasets['train_ruiz']
number_of_epochs = 500
k_folds = 5  # Number of folds for K-Fold Cross-Validation
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=24)

# List to store validation losses for each fold
fold_val_losses = []
# Classification threshold for AMP prediction:
threshold = 0.5
# Seed values for each fold to ensure reproducibility:
seeds = [0,3,9,1,21]

# Iterate over the K-Folds
for fold, (train_ids, val_ids) in enumerate(kfold.split(ruiz_training_datataset)):
    print(f'//////// Fold {fold+1}/{k_folds}: ///////////')
    print(f'  Train IDs: {len(train_ids)} samples')
    print(f'  Val IDs:   {len(val_ids)} samples')

    seed = seeds[fold]
    
    # Initialize the model and optimizer for the current fold:
    model, optimizer = initialize_model(initial_dim_gcn,
                                        edge_dim_feature,
                                        hidden_dim_nn_1,
                                        hidden_dim_nn_2,
                                        hidden_dim_gat_0,
                                        hidden_dim_fcn_1,
                                        hidden_dim_fcn_2,
                                        hidden_dim_fcn_3,
                                        dropout,
                                        
                                        learning_rate,
                                        weight_decay,
                                        seed = seed
                                        )

    # Create subsets and dataloaders for training and validation:
    train_subsampler = Subset(ruiz_training_datataset, train_ids)
    val_subsampler = Subset(ruiz_training_datataset, val_ids)
    tun_train_dataloader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    tun_val_dataloader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=True)
    
    # Lists to store losses for training and validation:
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') # Initialize best validation loss with infinity
    best_model_fold = None
    
    # Entrenamiento y validación por épocas
    for epoch in range(1, number_of_epochs+1):
        print(f'  Epoch {epoch}/{number_of_epochs}:')
        
        # Training step:
        model.train()
        train_loss = amp_train(model, device, tun_train_dataloader, optimizer, epoch)
        train_losses.append(train_loss)
        
        # Validation step:
        model.eval()
        val_loss = amp_validation(model, device, tun_val_dataloader, epoch)
        val_losses.append(val_loss)
        
        # Save the best model based on validation loss:
        if val_loss < best_val_loss:
            best_val_loss = val_loss # Update best validation loss
            best_model_fold = model.state_dict() # Save the best model weights
            best_fold_index = fold  # Update the fold index for the best model
    
    # Save the best model weights for the current fold:
    torch.save(best_model_fold, f"weights/AMP/best_model_weights_fold_{fold+1}.pth")
    print(f'  Best model in this fold: best_model_weights_fold_{fold+1}.pth')
    
    # Store the best validation loss for the current fold:
    fold_val_losses.append((fold, best_val_loss))
    print(f'  Lowest validation loss in this fold: {best_val_loss:.4f}\n')
    
    # Plot the loss curves for the current fold:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training loss', color='darkorange')
    plt.plot(val_losses, label='Validation loss', color='seagreen')
    plt.title(f'Training and Validation Loss\nFold {fold+1}\nBest Validation Loss: {best_val_loss:.5f}', fontsize=17)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize=14)
    plt.savefig(f'results/AMP/loss_curve_fold_{fold+1}.png', dpi=216)
    plt.show()
    
    # Importing the best model weights file path for reference:
    weights_file = f"weights/AMP/best_model_weights_fold_{fold+1}.pth"
    
    # ------------------------------------////////// Training set /////////////---------------------------------------------------
    # Predict on the training set using the saved model from the current fold
    training_sequences, training_target, training_logits, training_pred_csv, training_scores = amp_predict_test(
                                                                                                    model,
                                                                                                    tun_train_dataloader,
                                                                                                    device,
                                                                                                    weights_file,  # Use the saved model from the current fold
                                                                                                    threshold,
                                                                                                    has_targets = True
                                                                                                    )
    
    # Save a CSV file with the prediction values for the training set
    prediction_train_set = {
                            'Sequence': training_sequences,
                            'Target': training_target,
                            'Scores': training_scores,
                            'Prediction': training_pred_csv
                            }
    
    # Convert the predictions into a DataFrame and save as an Excel file
    df = pd.DataFrame(prediction_train_set)
    df.to_excel(f'results/AMP/training_prediction_fold_{fold+1}.xlsx', index=False)
    
    # Evaluate the model on the training set and obtain metrics
    TP_training, TN_training, FP_training, FN_training, ACC_training, PR_training, \
    SN_training, SP_training, F1_training, mcc_training, roc_auc_training, ap_training = \
    amp_evaluate_model(
                        prediction=training_scores,
                        target=training_target,
                        dataset_type=f'Training_fold_{fold+1}',
                        threshold=threshold,
                        device=device
                        
                        )
    
    # -------------------------------------------- ////////// Validation Set //////////-------------------------------------------------
    # Predict on the validation set using the saved model from the current fold
    validation_sequences, validation_target, validation_pred, validation_pred_csv, validation_scores = amp_predict_test(
                                                                                                                        model,
                                                                                                                        tun_val_dataloader,
                                                                                                                        device,
                                                                                                                        weights_file,  # Use the saved model from the current fold
                                                                                                                        threshold,
                                                                                                                        has_targets = True
                                                                                                                        )
    
    # Save a CSV file with the prediction values for the validation set
    prediction_validation_set = {
                                'Sequence': validation_sequences,
                                'Target': validation_target,
                                'Scores': validation_scores,
                                'Prediction': validation_pred_csv
                                }
    
    df = pd.DataFrame(prediction_validation_set)
    df.to_excel(f'results/AMP/validation_prediction_fold_{fold+1}.xlsx', index=False)
    
    # Evaluate the model on the validation set and obtain metrics
    TP_validation, TN_validation, FP_validation, FN_validation, ACC_validation, PR_validation, \
    SN_validation, SP_validation, F1_validation, mcc_validation, roc_auc_validation, ap_validation = \
    amp_evaluate_model( 
                        prediction= validation_scores,
                        target=validation_target,
                        dataset_type=f'Validation_fold_{fold+1}',
                        threshold=threshold,
                        device=device
                        )
    
    # Print metrics for the training set in this fold
    print(f'///// Fold {fold+1}-Training Metrics: //////')
    print(f'  Accuracy: {ACC_training:.4f}')
    print(f'  Precision: {PR_training:.4f}')
    print(f'  Recall: {SN_training:.4f}')
    print(f'  F1-score: {F1_training:.4f}')
    print(f'  MCC: {mcc_training:.4f}')
    print(f'  ROC AUC: {roc_auc_training:.4f}')
    print(f'  Average Precision: {ap_training:.4f}\n')
    
    # Print metrics for the validation set in this fold
    print(f'///// Fold {fold+1}-Validation Metrics: /////')
    print(f'  Accuracy: {ACC_validation:.4f}')
    print(f'  Precision: {PR_validation:.4f}')
    print(f'  Recall: {SN_validation:.4f}')
    print(f'  F1-score: {F1_validation:.4f}')
    print(f'  MCC: {mcc_validation:.4f}')
    print(f'  ROC AUC: {roc_auc_validation:.4f}')
    print(f'  Average Precision: {ap_validation:.4f}\n')


finish_time_training = time.time()
time_training= (finish_time_training - start_time) / 60 

# Extract only the validation losses from the list of tuples
val_losses = [val_loss for _, val_loss in fold_val_losses]

# Calculate the average and standard deviation of the validation losses
avg_val_loss = np.mean(val_losses)
std_val_loss = np.std(val_losses)

# Display the average and standard deviation of the validation losses across all folds
print(f'Average validation loss across all folds: {avg_val_loss:.7f} ± {std_val_loss:.7f}')

# --------------------------------------------////////// Test Set //////////---------------------------------------------------
print('//////// Testing ///////')

# Make predictions using the models from all five folds
folds = [1, 2, 3, 4, 5]
sequences, targets = [], []
all_scores = []
start_time_testing = time.time()

# Iterate over the folds to test each model
for fold in folds:
    weights_file = f"weights/AMP/best_model_weights_fold_{fold}.pth"
    
    # Make predictions on the test set using the loaded model
    test_sequences, test_target, _, test_pred_csv, test_scores = amp_predict_test(
                                                                                    model,
                                                                                    ruiz_test_dataloader,
                                                                                    device,
                                                                                    weights_file,
                                                                                    threshold,
                                                                                    has_targets = True
                                                                                )
    
    # Store the sequences and targets only once (from the first fold)
    if not sequences:
        sequences = test_sequences
        targets = test_target
        
    # Store the scores of the model for each fold
    all_scores.append(test_scores)

# Convert the predictions into a DataFrame with scores from all folds
df = pd.DataFrame({
    'Sequence': sequences,
    'Target': targets,
    'Scores model 1': all_scores[0],
    'Scores model 2': all_scores[1],
    'Scores model 3': all_scores[2],
    'Scores model 4': all_scores[3],
    'Scores model 5': all_scores[4],
})

# Calculate the average and standard deviation of predictions across the folds
df['Average score'] = df[['Scores model 1', 'Scores model 2', 'Scores model 3', 'Scores model 4', 'Scores model 5']].mean(axis=1)
df['Standard deviation'] = df[['Scores model 1', 'Scores model 2', 'Scores model 3', 'Scores model 4', 'Scores model 5']].std(axis=1)

# Generate final predictions by rounding the average score based on the threshold
df['Prediction'] = (df['Average score'] >= threshold).astype(int)

# Save the test predictions as an Excel file
df.to_excel('results/AMP/testing_prediction.xlsx', index=False)

finish_time_testing = time.time()
time_prediction = (finish_time_testing - start_time_testing) / 60

# Calculate evaluation metrics for the test set
TP_test, TN_test, FP_test, FN_test, ACC_test, PR_test, \
SN_test, SP_test, F1_test, mcc_test, roc_auc_test, ap_test = amp_evaluate_model(
                                                                        prediction=df['Average score'].values,
                                                                        target=df['Target'].values,
                                                                        dataset_type='Testing',
                                                                        threshold=threshold,
                                                                        device=device
                                                                    )
# Print the test set evaluation metrics
print(f'///// Testing Metrics: //////')
print(f'  Accuracy: {ACC_test:.4f}')
print(f'  Precision: {PR_test:.4f}')
print(f'  Recall: {SN_test:.4f}')
print(f'  F1-score: {F1_test:.4f}')
print(f'  MCC: {mcc_test:.4f}')
print(f'  ROC AUC: {roc_auc_test:.4f}')
print(f'  Average Precision: {ap_test:.4f}\n')

finish_time = time.time()
total_time = (finish_time - start_time) / 60

#--------------------------------///////////Result DataFrame////////////---------------------------------------
data = {
    "Metric": [
    "node_features",
    "edge_features",
    "initial_dim_gcn",
    "edge_dim_feature",
    "hidden_dim_nn_1",
    "hidden_dim_nn_2",
    "hidden_dim_gat_0",
    "hidden_dim_fcn_1",
    "hidden_dim_fcn_2",
    "hidden_dim_fcn_3",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "number_of_epochs",
    "threshold",
    "TP_training",
    "TN_training",
    "FP_training",
    "FN_training",
    "ACC_training",
    "PR_training",
    "SN_training",
    "SP_training",
    "F1_training",
    "mcc_training",
    "roc_auc_training",
    "TP_validation",
    "TN_validation",
    "FP_validation",
    "FN_validation",
    "ACC_validation",
    "PR_validation",
    "SN_validation",
    "SP_validation",
    "F1_validation",
    "mcc_validation",
    "roc_auc_validation",
    "TP_test",
    "TN_test",
    "FP_test",
    "FN_test",
    "ACC_test",
    "PR_test",
    "SN_test",
    "SP_test",
    "F1_test",
    "mcc_test",
    "roc_auc_test",
    "time_preprocessing",
    "time_training",
    "time_prediction",
    "total_time"
    ],
    "Value": [
        ruiz_testing_datataset.num_features,
        ruiz_testing_datataset.num_edge_features,
        initial_dim_gcn,
        edge_dim_feature ,
        hidden_dim_nn_1 ,
        hidden_dim_nn_2 ,
        hidden_dim_gat_0,
        hidden_dim_fcn_1 ,
        hidden_dim_fcn_2 ,
        hidden_dim_fcn_3 ,
        batch_size,
        learning_rate,
        weight_decay,
        number_of_epochs,
        threshold,
        TP_training,
        TN_training,
        FP_training,
        FN_training,
        ACC_training, 
        PR_training, 
        SN_training, 
        SP_training,
        F1_training,
        mcc_training,
        roc_auc_training,
        TP_validation,
        TN_validation,
        FP_validation,
        FN_validation,
        ACC_validation, 
        PR_validation, 
        SN_validation, 
        SP_validation,
        F1_validation,
        mcc_validation,
        roc_auc_validation,
        TP_test,
        TN_test,
        FP_test,
        FN_test,
        ACC_test, 
        PR_test, 
        SN_test, 
        SP_test,
        F1_test,
        mcc_test,
        roc_auc_test,
        time_preprocessing, 
        time_training,
        time_prediction,
        total_time
    ],
    
}


df = pd.DataFrame(data)
df.to_csv('results/AMP/results_training_validation_test.csv', index=False)

print('/////////////// Ready ////////////////////')


# %%
