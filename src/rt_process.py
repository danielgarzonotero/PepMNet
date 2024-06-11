import torch


def rt_train(model, device, dataloader, optim, epoch):
    
    model.train()
    
    loss_func = torch.nn.MSELoss(reduction='sum') 
    loss_collect = 0
    
    # Looping over the dataloader allows us to pull out or input/output data:
    # Enumerate allows us to also get the batch number:
    for batch in dataloader:
        batch = batch.to(device)
        
        # Zero out the optimizer:        
        optim.zero_grad()
        
        # Make a prediction:
        pred = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                    batch.cc,
                    batch.monomer_labels,
                    batch.aminoacids_features,
                    batch.amino_index
                    )
        
        # Calculate the loss:
        loss = loss_func(pred.double(), batch.y.double())
        
        # Backpropagation:
        loss.backward()
        optim.step()
        
        # Calculate the loss and add it to our total loss
        loss_collect += loss.item()  # loss summed across the batch
        
        # Print out our Training loss so we know how things are going

    # Return our normalized losses so we can analyze them later:
    loss_collect /= len(dataloader.dataset)
    
    print(
        "Epoch:{}   Training dataset:   Loss per Datapoint: {:.3f}%".format(
            epoch, loss_collect*100
        )
    )  
    return loss_collect    


def rt_validation(model, device, dataloader, epoch):

    model.eval()
    loss_collect = 0
    loss_func = torch.nn.MSELoss(reduction='sum') 
    
    # Remove gradients:
    with torch.no_grad():

        # Looping over the dataloader allows us to pull out or input/output data:
        for batch in dataloader:
            batch = batch.to(device)
            
            # Make a prediction:
            pred = model( 
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch,
                        batch.cc,
                        batch.monomer_labels,
                        batch.aminoacids_features,
                        batch.amino_index
                        )
            
            # Calculate the loss:
            loss = loss_func(pred.double(), batch.y.double())
            # Calculate the loss and add it to our total loss
            loss_collect += loss.item()  # loss summed across the batch

    loss_collect /= len(dataloader.dataset)
    
    # Print out our test loss so we know how things are going
    print(
        "Epoch:{}   Validation dataset: Loss per Datapoint: {:.3f}%".format(
            epoch, loss_collect*100
        )
    )  
    print('---------------------------------------')   
    
    # Return our normalized losses so we can analyze them later:
    return loss_collect


def rt_predict_test(model, dataloader, device, weights_file):
    
    model.eval()
    model.load_state_dict(torch.load(weights_file))
    
    x_all = []
    y_all = []
    pred_all = []
    
    # Remove gradients:
    with torch.no_grad():
        
        # Looping over the dataloader allows us to pull out or input/output data:
        for batch in dataloader:
            batch = batch.to(device)
            # Make a prediction:
            pred = model(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch,
                        batch.cc,
                        batch.monomer_labels,
                        batch.aminoacids_features,
                        batch.amino_index
                        )
            
            x_all.extend(batch.sequence)
            y_all.append(batch.y.to(device))
            pred_all.append(pred.to(device))

    
    y_all = torch.concat(y_all)
    pred_all = torch.concat(pred_all)
    
    return x_all, y_all, pred_all

def rt_tester(model, dataloader, device, weights_file):
    
    model.eval()
    model.load_state_dict(torch.load(weights_file))
    
    x_all = []
    pred_all = []
    
    # Remove gradients:
    with torch.no_grad():
        
        # Looping over the dataloader allows us to pull out or input/output data:
        for batch in dataloader:
            batch = batch.to(device)
            # Make a prediction:
            pred = model(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch,
                        batch.cc,
                        batch.monomer_labels,
                        batch.aminoacids_features,
                        batch.amino_index
                        )
            
            x_all.extend(batch.sequence)
            pred_all.append(pred.to(device))

    pred_all = torch.concat(pred_all)
    
    return x_all, pred_all

