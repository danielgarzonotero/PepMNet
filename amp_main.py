#%%
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.explain import CaptumExplainer, Explainer

import pandas as pd
from src.device import device_info
from src.data import GeoDataset_1
from src.amp_model import amp_GCN_Geo
from src.amp_process import amp_train, amp_validation, amp_predict_test
from src.amp_evaluation_metrics import amp_evaluate_model

device_information = device_info()
print(device_information)
device = device_information.device

start_time = time.time()

## SET UP DATALOADERS: 

# Build starting dataset: 
datasets = {
            'training_dataset': GeoDataset_1(raw_name='data/AMP/dataset Chung/Chung_Xiao_train_AB_6153.csv',
                                            root='',
                                            index_x=0,
                                            index_y=1,
                                            ),
            'validation_dataset': GeoDataset_1(raw_name='data/AMP/dataset Chung/Chung_Xiao_val_AB_1640.csv',
                                                root='',
                                                index_x=0,
                                                index_y=1
                                            ),
            'testing_dataset': GeoDataset_1(raw_name='data/AMP/dataset Chung/Chung_Xiao_test_AB_410.csv',
                                            root='',
                                            index_x=0,
                                            index_y=1
                                            ),
            'fold_1': GeoDataset_1(raw_name='data/AMP/dataset Ruiz/Fold1_clean.csv',
                                            root='',
                                            index_x=0,
                                            index_y=2,
                                            ),
            'fold_2': GeoDataset_1(raw_name='data/AMP/dataset Ruiz/Fold2_clean.csv',
                                            root='',
                                            index_x=0,
                                            index_y=2
                                            ),
            'fold_3': GeoDataset_1(raw_name='data/AMP/dataset Ruiz/Fold3_clean.csv',
                                            root='',
                                            index_x=0,
                                            index_y=2
                                            ),
            'fold_4': GeoDataset_1(raw_name='data/AMP/dataset Ruiz/Fold4_clean.csv',
                                            root='',
                                            index_x=0,
                                            index_y=2
                                            ),
            'test_ruiz': GeoDataset_1(raw_name='data/AMP/dataset Ruiz/Test_clean.csv',
                                            root='',
                                            index_x=0,
                                            index_y=1
                                            ),
            'train_ruiz': GeoDataset_1(raw_name='data/AMP/dataset Ruiz/Train_clean.csv',
                                            root='',
                                            index_x=0,
                                            index_y=1
                                            )
            }


training_datataset = datasets['training_dataset']
validation_datataset = datasets['validation_dataset']
testing_datataset = datasets['testing_dataset']

ruiz_training_datataset = datasets['train_ruiz']
ruiz_testing_datataset = datasets['test_ruiz']

#  Number of datapoints in each dataset:
#Pretraining
size_training_dataset = 0.9
size_validation_dataset = 1-size_training_dataset
n_training = int(len(ruiz_training_datataset ) * size_training_dataset)
n_validation = len(ruiz_training_datataset ) - n_training

ruiz_training_set, ruiz_validation_set = torch.utils.data.random_split(ruiz_training_datataset  , [n_training, n_validation], generator=torch.Generator().manual_seed(24))

print('Number of NODES features: ', training_datataset.num_features)
print('Number of EDGES features: ', training_datataset.num_edge_features)

finish_time_preprocessing = time.time()
time_preprocessing = (finish_time_preprocessing - start_time) / 60 

# Define dataloaders para conjuntos de entrenamiento, validación y prueba:
batch_size = 4000  #TODO
train_dataloader = DataLoader(training_datataset, batch_size, shuffle=True)
val_dataloader = DataLoader(validation_datataset , batch_size, shuffle=True)
test_dataloader = DataLoader(testing_datataset, batch_size, shuffle=True)

ruiz_train_dataloader = DataLoader(ruiz_training_set, batch_size, shuffle=True)
ruiz_val_dataloader = DataLoader(ruiz_validation_set, batch_size, shuffle=True)
ruiz_test_dataloader = DataLoader(ruiz_testing_datataset , batch_size, shuffle=True)


## RUN TRAINING LOOP: 

# Train with a random seed to initialize weights:
torch.manual_seed(24)

# Set up model:
# Initial Inputs
initial_dim_gcn = training_datataset.num_features
edge_dim_feature = training_datataset.num_edge_features

hidden_dim_nn_1 = 20
hidden_dim_nn_2 = 10

hidden_dim_gat_0 = 10

hidden_dim_fcn_1 = 100
hidden_dim_fcn_2 = 50
hidden_dim_fcn_3 = 10
dropout = 0.2


model = amp_GCN_Geo(
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



#/////////////////// Training /////////////////////////////
# Set up optimizer:
learning_rate = 1E-3 #TODO 
weight_decay = 1E-4 
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# Definir el scheduler ReduceLROnPlateau
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold= 0.1, verbose= True, mode='max', patience=100, factor=0.1)


train_losses = []
val_losses = []

best_val_loss = float('inf')  # infinito

start_time_training = time.time()
number_of_epochs = 500

for epoch in range(1, number_of_epochs+1):
    train_loss = amp_train( model,
                        device,
                        ruiz_train_dataloader ,
                        optimizer,
                        epoch,
                        type_dataset='training')

    train_losses.append(train_loss)

    val_loss = amp_validation(  model,
                            device,
                            ruiz_val_dataloader,
                            epoch,
                            type_dataset='validation')
    val_losses.append(val_loss)

    # Programar el LR basado en la pérdida de validación
    #scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "weights_AMP/best_model_weights.pth")

finish_time_training = time.time()
time_training = (finish_time_training - start_time_training) / 60


#---------------------------------------//////// Losse curves ///////// ---------------------------------------------------------

plt.plot(train_losses, label='Training loss', color='darkorange') 
plt.plot(val_losses, label='Validation loss', color='seagreen')  

# Agregar texto para la mejor pérdida de validación
best_val_loss_epoch = val_losses.index(best_val_loss)  # Calcular el epoch correspondiente a la mejor pérdida de validación
best_val_loss = best_val_loss
# Añadir la época y el mejor valor de pérdida como subtítulo
plt.title('Training and Validation Loss\nAMP Dataset\nBest Validation Loss: Epoch {}, Value {:.4f}'.format(best_val_loss_epoch, best_val_loss), fontsize=17)
# Aumentar el tamaño de la fuente en la leyenda
plt.legend(fontsize=14) 
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Guardar la figura en formato PNG con dpi 216
plt.savefig('results/loss_curve.png', dpi=216)
plt.show()

# Testing:
weights_file = "weights/best_model_weights.pth"
threshold = 0.5


# ------------------------------------////////// Training set /////////////---------------------------------------------------


training_input, training_target, training_pred, training_pred_csv, trainig_scores = amp_predict_test(   model,
                                                                                                ruiz_train_dataloader,
                                                                                                device,
                                                                                                weights_file,
                                                                                                threshold,
                                                                                                type_dataset='training')

#Saving a CSV file with prediction values
prediction_train_set = {
                    'Sequence':training_input,
                    'Target': training_target.cpu().numpy().T.flatten().tolist(),
                    'Scores' : trainig_scores,
                    'Prediction':  training_pred_csv
                    
                    }

df = pd.DataFrame(prediction_train_set)
df.to_excel('results/training_prediction.xlsx', index=False)

# Evaluation metrics:

TP_training, TN_training, FP_training, FN_training, ACC_training, PR_training, \
SN_training, SP_training, F1_training, mcc_training, roc_auc_training = \
amp_evaluate_model( prediction=training_pred,
            target=training_target,
            dataset_type='Training',
            threshold=threshold,
            device=device)


#-------------------------------------------- ////////// Validation Set //////////-------------------------------------------------
validation_input, validation_target, validation_pred, validation_pred_csv, validation_scores = amp_predict_test(model,
                                                                                                        ruiz_val_dataloader,
                                                                                                        device, weights_file, threshold, type_dataset='validation')

#Saving a CSV file with prediction values
prediction_validation_set = {
                        'Sequence':validation_input,
                        'Target': validation_target.cpu().numpy().T.flatten().tolist(),
                        'Scores' : validation_scores,
                        'Prediction':  validation_pred_csv
                        }

df = pd.DataFrame(prediction_validation_set)
df.to_excel('results/validation_prediction.xlsx', index=False)


# Evaluation metrics:

TP_validation, TN_validation, FP_validation, FN_validation, ACC_validation, PR_validation, \
SN_validation, SP_validation, F1_validation, mcc_validation, roc_auc_validation = \
amp_evaluate_model( prediction = validation_pred,
            target = validation_target,
            dataset_type = 'Validation',
            threshold = threshold,
            device = device)

# --------------------------------------------////////// Test Set //////////---------------------------------------------------
start_time_testing = time.time()

test_input, test_target, test_pred, test_pred_csv, testing_scores = amp_predict_test(model, ruiz_test_dataloader, device, weights_file,threshold, type_dataset='testing')

finish_time_testing = time.time()
time_prediction = (finish_time_testing- start_time_testing) / 60

#Saving a CSV file with prediction values
prediction_test_set = {
                    'Sequence':test_input,
                    'Target': test_target.cpu().numpy().T.flatten().tolist(),
                    'Scores' : testing_scores,
                    'Prediction': test_pred_csv
                    }

df = pd.DataFrame(prediction_test_set)
df.to_excel('results/testing_prediction.xlsx', index=False)

# Evaluation metrics:

TP_test, TN_test, FP_test, FN_test, ACC_test, PR_test, \
SN_test, SP_test, F1_test, mcc_test, roc_auc_test = \
amp_evaluate_model( prediction=test_pred,
            target = test_target,
            dataset_type = 'Testing',
            threshold = threshold,
            device = device)


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
    training_datataset.num_features,
    training_datataset.num_edge_features,
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
df.to_csv('results/results_training_validation_test.csv', index=False)

print('/////////////// Ready ////////////////////')
#-------------------------------------///////// Explainer ////// -------------------------------------------

''' explainer = Explainer(
                    model=model,
                    algorithm=CaptumExplainer('IntegratedGradients'),
                    explanation_type='model', #Explains the model prediction.
                    model_config=dict(
                        mode='regression', #or 
                        task_level='node', #ok
                        return_type='raw', #ok
                    ),
                    node_mask_type='attributes', #"attributes": Will mask each feature across all nodes
                    edge_mask_type=None,
                    threshold_config=dict(
                        threshold_type='hard', #The type of threshold to apply. 
                        value = 0  #The value to use when thresholding.
                    ),
                    )


# Generar explicaciones para cada nodo en cada lote del DataLoader
aminoacids_features_dict = torch.load('data/dataset/dictionaries/training/aminoacids_features_dict.pt', map_location=device)
blosum62_dict = torch.load('data//dataset/dictionaries/training/blosum62_dict.pt', map_location=device)

batch_size = len(training_datataset.data.cc)
train_dataloader = DataLoader(training_datataset, batch_size, shuffle=True)

start_time = time.time()

for batch in train_dataloader:
    batch = batch.to(device)
    x, target, edge_index,  edge_attr, idx_batch, cc, monomer_labels, amino = batch.x, batch.y, batch.edge_index, batch.edge_attr, batch.batch, batch.cc, batch.monomer_labels, batch.aminoacids_features
    
    node_features_dim = x.size(-1)
    print("Número de características de los nodos:", node_features_dim)
    
    explanation = explainer(
                            x=x, 
                            target = target,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            aminoacids_features_dict=aminoacids_features_dict,
                            blosum62_dict=blosum62_dict,
                            idx_batch=idx_batch,
                            cc=cc,
                            monomer_labels=monomer_labels
                        )
    
    path = 'results/IntegratedGradients_feature_importance.png'
    explanation.visualize_feature_importance(path, top_k = node_features_dim) 
    finish_time = time.time()
    time_prediction = (finish_time- start_time) / 60
    print('\nTime Feature Importance:',time_prediction , 'min')
 '''

# %%
