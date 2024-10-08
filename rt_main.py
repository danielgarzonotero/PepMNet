
#%%
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from math import sqrt
from src.rt_process import rt_train, rt_test, rt_predict_test
from src.rt_model import rt_pepmnet 
from src.device import device_info
from src.data import   GeoDataset_1
import os

# Get information about the device (GPU/CPU)
device_information = device_info()
print(device_information)
# Define the device (GPU or CPU) to be used
device = device_information.device

start_time = time.time()

# Define the dataset name
#1) 'hela_mod3'
#2) 'yeast_unmod'
#3) 'scx'
#4) 'luna_hilic'
#5) 'luna_silica'
#6) 'atlantis_silica'
#7) 'xbridge_amide'
#8) 'misc_dia'

datasets = [
            'hela_mod3',
            'yeast_unmod',
            'scx',
            'luna_hilic',
            'luna_silica',
            'atlantis_silica',
            'xbridge_amide',
            'misc_dia',
            ]

# Iterate over each dataset
for name_dataset in datasets:
    if name_dataset in ['hela_mod', 'hela_mod3','yeast_unmod', 'misc_dia']:
        time_unit = 's'
    else:
        time_unit = 'min'
    
    # Clear the GPU cache
    torch.cuda.empty_cache()
    
    # ////////////// Dataset Initialization and Splitting //////////////:
    
    # Load/Processing the dataset using GeoDataset_1, specifying the file paths for train and test datasets
    # raw_name: Path to the CSV file containing the data
    # index_x: The column index for the sequences in the CSV file
    # index_y: The column index for the target property (RT of the sequence)
    # has_targets: Set to True because this is a supervised task with known targets (RT)  
    
    datasets = {
                name_dataset: GeoDataset_1(
                                            raw_name=f'data/RT/{name_dataset}.csv',
                                            root='',
                                            index_x=0,
                                            index_y=1,
                                            has_targets=True
                                            )
                }
    
    # Split the dataset into training and testing sets with a specified ratio
    training_testing_datataset = datasets[name_dataset]
    
    size_training_dataset = 0.90
    size_testing_dataset = 1 - size_training_dataset
    n_training = int(len(training_testing_datataset) * size_training_dataset)
    n_validation = len(training_testing_datataset) - n_training
    
    # Perform random split of the dataset
    training_set, testing_set = torch.utils.data.random_split(training_testing_datataset, [n_training, n_validation], generator=torch.Generator().manual_seed(42))
    
    # Create DataLoader objects for training and validation.
    batch_size = 25
    train_dataloader = DataLoader(training_set, batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_set, batch_size, shuffle=True)
    
    # Print Features Information:
    print('Number of NODES features: ', training_testing_datataset.num_features)
    print('Number of EDGES features: ', training_testing_datataset.num_edge_features)
    
    finish_time_preprocessing = time.time()
    time_preprocessing = (finish_time_preprocessing - start_time) / 60  
    
    # Set seed for reproducibility
    torch.manual_seed(0)
    
    # ////////////// Model Setup //////////////:
    # Define the architecture and parameters of the model
    initial_dim_gcn = training_testing_datataset.num_features
    edge_dim_feature = training_testing_datataset.num_edge_features
    
    hidden_dim_nn_1 = 500
    hidden_dim_nn_2 = 250  
    hidden_dim_nn_3 = 100
    
    hidden_dim_gat_0 = 15
    
    hidden_dim_fcn_1 = 100
    hidden_dim_fcn_2 = 50
    hidden_dim_fcn_3 = 10
    
    dropout = 0
    
    # Instantiate the rt_pepmnet model with the defined parameters
    model = rt_pepmnet(
                        initial_dim_gcn,
                        edge_dim_feature,
                        hidden_dim_nn_1,
                        hidden_dim_nn_2,
                        hidden_dim_nn_3,
                        hidden_dim_gat_0,
                        hidden_dim_fcn_1,
                        hidden_dim_fcn_2,
                        hidden_dim_fcn_3,
                        dropout
                        ).to(device)
    
    # ////////////// Optimizer Setup //////////////:
    learning_rate = 1E-3 
    optimizer = optim.Adam(model.parameters(), learning_rate)
    
    train_losses = []
    test_losses = []
    best_val_loss = float('inf')  # Initialize best training loss to infinity
    
    start_time_training = time.time()
    number_of_epochs = 100
    
    # ////////////// Training Loop ////////////// :
    
    print("\n/////// Training - {} ////////".format(name_dataset))
    # Loop over the number of epochs
    for epoch in range(1, number_of_epochs+1):
        
        # Train the model for one epoch and calculate training loss.
        train_loss = rt_train(
                                model,
                                device,
                                train_dataloader,
                                optimizer,
                                epoch
                                )
        
        train_losses.append(train_loss)
        
        # Testing the model on the test set and calculate test loss.
        test_loss = rt_test(
                                model,
                                device,
                                test_dataloader,
                                epoch
                                )
        
        test_losses.append(test_loss)
        
        # Save the model's weights if the current training loss is the best so far.
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            torch.save(model.state_dict(), "weights/RT/{}_best_model_weights.pth".format(name_dataset))
    
    
    finish_time_training = time.time()
    time_training = (finish_time_training - start_time_training) / 60
    
    # ////////////// Evaluation Section //////////////:
    
    # Plotting Loss Curves:
    # Configure font and plot training and validation loss curves. 
    plt.plot(train_losses, label='Training Loss', color='darkorange', linewidth=2.5) 
    plt.plot(test_losses, label='Validation Loss', color='seagreen', linewidth=2.5)  
    plt.legend(fontsize=16) 
    plt.xlabel('Epochs', fontsize=16, labelpad=10)
    plt.ylabel('Sum of Squared Errors ($s^2$)', fontsize=16, labelpad=10)
    plt.title('Training and Validation Loss\n{} Dataset'.format(name_dataset), fontsize=17, pad=30)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(f'results/RT/{name_dataset}_loss_curves.png', dpi=300, bbox_inches='tight')  # Saving the loss curves plot
    plt.show()
    
    # Retrieving Best Model Weights:
    weights_file = "weights/RT/{}_best_model_weights.pth".format(name_dataset)
    
    # ////////////// Training Set //////////////:
    # Obtain predictions on the training set using the best model.
    input_all_train, target_all_train, pred_all_train = rt_predict_test(model,
                                                                        train_dataloader,
                                                                        device,
                                                                        weights_file,
                                                                        has_targets=True
                                                                        )
    
    # Saving Predictions as Excel:
    # Save the predictions, targets, and sequences as an Excel file.
    prediction_train_set = {
                            'Sequence': input_all_train,
                            'Target': target_all_train.cpu().numpy(),
                            'Prediction': pred_all_train.cpu().numpy()
                            }
    
    # Create a DataFrame from the prediction set
    df = pd.DataFrame(prediction_train_set)
    df.to_excel('results/RT/{}_training_set_prediction.xlsx'.format(name_dataset), index=False)
    
    # Calculate evaluation metrics such as R-squared, Pearson correlation, MAE, MSE, and RMSE on the training set.
    r2_train = r2_score(target_all_train.cpu(), pred_all_train.cpu())
    r_train, _ = pearsonr(target_all_train.cpu(), pred_all_train.cpu()) 
    mae_train = mean_absolute_error(target_all_train.cpu(), pred_all_train.cpu())
    mse_train = mean_squared_error(target_all_train.cpu(), pred_all_train.cpu(), squared=False)
    rmse_train = sqrt(mse_train)
    
    # Visualization: Scatter Plot for Training Set:
    # Plot a scatter plot of true vs. predicted values on the training set. 
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(target_all_train.cpu(), pred_all_train.cpu(), alpha=0.3, color='steelblue')
    plt.plot([min(target_all_train.cpu()), max(target_all_train.cpu())], [min(target_all_train.cpu()), max(target_all_train.cpu())], color='steelblue', ls="-", alpha=0.7, linewidth=3.5)
    plt.xlim([min(target_all_train.cpu()), max(target_all_train.cpu())])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Scatter Plot Training Set\n{name_dataset} Dataset', fontsize=18, pad=30)
    plt.xlabel(f"True Retention Time ({time_unit})", fontsize=18, labelpad=10)
    plt.ylabel(f"Predicted Retention Time ({time_unit})", fontsize=18, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f'results/RT/{name_dataset}_scatter_training.png', format="png", dpi=300, bbox_inches='tight')
    plt.show()
    
    #////////////// Testing Set //////////////:
    # Predictions on Testing Set:
    
    # Obtain predictions on the validation set using the best model.
    input_all_test, target_all_test, pred_all_test = rt_predict_test(   model,
                                                                        test_dataloader,
                                                                        device,
                                                                        weights_file,
                                                                        has_targets=True
                                                                        )
    
    # Save the predictions, targets, and sequences of the validation set as an Excel file.
    prediction_validation_set = {
                                'Sequence': input_all_test,
                                'Target': target_all_test.cpu().numpy(),
                                'Prediction': pred_all_test.cpu().numpy()
                                }
    
    df = pd.DataFrame(prediction_validation_set)
    df.to_excel('results/RT/{}_test_set_prediction.xlsx'.format(name_dataset), index=False)
    
    # Calculate evaluation metrics such as R-squared, Pearson correlation, MAE, MSE, and RMSE on the test set.
    r2_test = r2_score(target_all_test.cpu(), pred_all_test.cpu())
    r_test, _ = pearsonr(target_all_test.cpu(), pred_all_test.cpu())
    mae_test = mean_absolute_error(target_all_test.cpu(), pred_all_test.cpu())
    mse_test = mean_squared_error(target_all_test.cpu(), pred_all_test.cpu(), squared=False)
    rmse_test = sqrt(mse_test)
    
    # Visualization: Scatter Plot for test Set:
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(target_all_test.cpu(), pred_all_test.cpu(), alpha=0.3, color='steelblue')
    plt.plot([min(target_all_test.cpu()), max(target_all_test.cpu())], [min(target_all_test.cpu()), max(target_all_test.cpu())], color='steelblue', ls="-", alpha=0.7, linewidth=3.5)
    plt.xlim([min(target_all_test.cpu()), max(target_all_test.cpu())])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Scatter Plot Test Set\n{name_dataset} Dataset', fontsize=18, pad=30)
    plt.xlabel(f"True Retention Time ({time_unit})", fontsize=14, labelpad=10)
    plt.ylabel(f"Predicted Retention Time ({time_unit})", fontsize=14, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(f'results/RT/{name_dataset}_scatter_test.png', format="png", dpi=300, bbox_inches='tight')
    plt.show()
    
    #Times
    finish_time = time.time()
    time_prediction = (finish_time - finish_time_training) / 60
    total_time = (finish_time - start_time) / 60
    print("\n //// Preprocessing time: {:3f} minutes ////".format(time_preprocessing))
    print("\n //// Training time: {:3f} minutes ////".format(time_training))
    print("\n //// Prediction time: {:3f} minutes ////".format(time_prediction))
    print("\n //// Total time: {:3f} minutes ////".format(total_time))
    
    # Result DataFrame
    data = {
        "Metric": [
            "number_features",
            "num_edge_features",
            "initial_dim_gcn ",
            "edge_dim_feature",
            "hidden_dim_nn_1 ",
            "hidden_dim_nn_2 ",
            "hidden_dim_nn_3 ",
            "hidden_dim_gat_0",
            "hidden_dim_fcn_1 ",
            "hidden_dim_fcn_2 ",
            "hidden_dim_fcn_3 ",
            "training_test_percentage %",
            "batch_size", 
            "learning_rate",
            "number_of_epochs",
            "r2_train",
            "r_train",
            "mae_train",
            "mse_train", 
            "rmse_train", 
            "r2_test",
            "r_test",
            "mae_test",
            "mse_test",
            "rmse_test",
            "time_preprocessing", 
            "time_training",
            "time_prediction",
            "total_time"
        ],
        "Value": [
            training_testing_datataset.num_features,
            training_testing_datataset.num_edge_features,
            initial_dim_gcn,
            edge_dim_feature ,
            hidden_dim_nn_1 ,
            hidden_dim_nn_2 ,
            hidden_dim_nn_3,
            hidden_dim_gat_0,
            hidden_dim_fcn_1 ,
            hidden_dim_fcn_2 ,
            hidden_dim_fcn_3 ,
            size_training_dataset*100,
            batch_size,
            learning_rate,
            number_of_epochs,
            r2_train, 
            r_train, 
            mae_train, 
            mse_train,
            rmse_train,
            r2_test,
            r_test,
            mae_test, 
            mse_test,
            rmse_test,
            time_preprocessing, 
            time_training,
            time_prediction,
            total_time
        ],
        
    }
    
    df = pd.DataFrame(data)
    df.to_csv('results/RT/{}_dataset_results.csv'.format(name_dataset), index=False)

print('////////////// Done //////////////')

# %%
