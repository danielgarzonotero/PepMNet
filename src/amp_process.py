
import torch
import torch.nn.functional as F
import numpy as np


def amp_train(model, device, dataloader, optim, epoch):
    model.train()
    
    loss_func = torch.nn.BCEWithLogitsLoss() 
    loss_collect = 0
    
    # Looping over the dataloader allows us to pull out input/output data:
    for batch in dataloader:
        # Zero out the optimizer:        
        optim.zero_grad()
        batch = batch.to(device)

        # Make a prediction:
        pred = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    idx_batch=batch.batch,
                    cc=batch.cc,
                    monomer_labels=batch.monomer_labels,
                    aminoacids_features=batch.aminoacids_features,
                    amino_index= batch.amino_index
                    )
        
        # Calculate the loss:
        loss = loss_func(pred.double(), batch.y.double())
        
        # Backpropagation:
        loss.backward()
        optim.step()
        
        # Calculate the loss and add it to our total loss
        loss_collect += loss.item()  # loss summed across the batch
        
    # Return our normalized losses so we can analyze them later:
    loss_collect /= len(dataloader.dataset)
    
    print(
        "Epoch:{}   Training dataset:   Loss per Datapoint: {:.4f}%".format(
            epoch, loss_collect * 100
        )
    ) 
    return loss_collect    

def amp_validation(model, device, dataloader, epoch):

    model.eval()
    loss_collect = 0
    loss_func = torch.nn.BCEWithLogitsLoss() 
    
    # Remove gradients:
    with torch.no_grad():
        
        for batch in dataloader:
            
            batch = batch.to(device)
            
            # Make a prediction:
            pred = model(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        idx_batch=batch.batch,
                        cc=batch.cc,
                        monomer_labels=batch.monomer_labels,
                        aminoacids_features=batch.aminoacids_features,
                        amino_index= batch.amino_index
                            )
            
            # Calculate the loss:
            loss = loss_func(pred.double(), batch.y.double())  
            
            # Calculate the loss and add it to our total loss
            loss_collect += loss.item()  # loss summed across the batch
            
    loss_collect /= len(dataloader.dataset)
    
    # Print out our test loss so we know how things are going
    print(
        "Epoch:{}   Validation dataset: Loss per Datapoint: {:.4f}%".format(
            epoch, loss_collect * 100
        )
    )  
    print('---------------------------------------')     
    # Return our normalized losses so we can analyze them later:
    return loss_collect


def amp_predict_test(model, dataloader, device, weights_file, threshold, has_targets=True):
    model.eval()
    model.load_state_dict(torch.load(weights_file, map_location=device))
    
    sequences = []
    logits = []
    scores = []
    targets = [] if has_targets else None  # Initialize targets only if needed
    
    # Remove gradients:
    with torch.no_grad():
        # Loop over the dataloader to pull out input/output data:
        for batch in dataloader:
            batch = batch.to(device)
            
            # Make a prediction:
            pred = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                idx_batch=batch.batch,
                cc=batch.cc,
                monomer_labels=batch.monomer_labels,
                aminoacids_features=batch.aminoacids_features,
                amino_index= batch.amino_index
            )
            score = torch.sigmoid(pred)  # Convert to probabilities
            
            sequences.extend(batch.sequence)
            logits.append(pred)
            scores.append(score)
            
            if has_targets:
                targets.extend(batch.y.cpu().numpy())
                
    # Concatenate the lists of tensors into a single tensor
    logits = torch.cat(logits, dim=0)
    
    # Export the prediction rounded based on the threshold
    scores = torch.cat(scores, dim=0).tolist()
    scores = [round(s, 3) for s in scores]  # Round to 3 decimal places
    predictions = [custom_round(pred, threshold) for pred in scores]
    
    if has_targets:
        return sequences, targets, logits, predictions, scores
    else:
        return sequences, logits, predictions, scores


def custom_round(pred, threshold):
    return 1 if pred >= threshold else 0
