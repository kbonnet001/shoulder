import numpy as np
import torch
import matplotlib as plt
import matplotlib.pyplot as plt

def mean_distance(predictions, targets):
    """
    Compute mean distance beetween predictions and targets

    INPUTS :
    - predictions (torch.Tensor): Model's predictions 
    - targets (torch.Tensor): Targets

    OUPUT : 
        float: mean distance
    """
    distance = torch.mean(torch.abs(predictions - targets))
    return distance.item()


def train(model, train_loader, optimizer, criterion, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Train the model on the training dataset.

    INPUT :
    - model (nn.Module): The neural network model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset, which provides batches of data.
    - optimizer (torch.optim.Optimizer): The optimization algorithm to update the model's weights.
    - criterion (torch.nn.Module): The loss function to minimize during training.
    - device (torch.device, optional): The device on which to perform computations (CPU or CUDA). Default is CUDA if available.

    OUTPUT :
    - epoch_loss (float): The average loss over the training dataset for the current epoch.
    - epoch_acc (float): The accuracy of the model on the training dataset for the current epoch.
    """

    model.train()  # Train mode
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # Gradients initialization
        outputs = model(inputs)  # Model prediction

        targets = targets.unsqueeze(1)
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Updating weights

        # Stats for plot
        running_loss += loss.item() * inputs.size(0)
        all_predictions.append(outputs)
        all_targets.append(targets)

    # Calculation of average loss
    epoch_loss = running_loss / len(train_loader.dataset)

    # Calculation of mean distance
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    # epoch_acc = accuracy(all_predictions, all_targets)
    epoch_acc = mean_distance(all_predictions, all_targets)

    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Evaluate the model on the validation or test dataset.

    INPUT :
    - model (nn.Module): The neural network model to be evaluated.
    - data_loader (DataLoader): DataLoader for the validation or test dataset, which provides batches of data.
    - criterion (torch.nn.Module): The loss function to calculate the error between the model predictions and true values.
    - device (torch.device, optional): The device on which to perform computations (CPU or CUDA). Default is CUDA if available.

    OUTPUT :
    - epoch_loss (float): The average loss over the validation or test dataset.
    - epoch_acc (float): The accuracy of the model on the validation or test dataset.
    """

    model.eval()  # Eval mode
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # Disable gradient calculation to speed up evaluation
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # Model prediction

            targets = targets.unsqueeze(1)
            loss = criterion(outputs, targets)  # Compute loss

            # Stats for plot
            running_loss += loss.item() * inputs.size(0)
            all_predictions.append(outputs)
            all_targets.append(targets)

    # Calculation of average loss
    epoch_loss = running_loss / len(data_loader.dataset)

    # Calculation of mean distance
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    # epoch_acc = accuracy(all_predictions, all_targets)
    epoch_acc = mean_distance(all_predictions, all_targets)

    return epoch_loss, epoch_acc

# Fonction pour entraîner et évaluer le modèle
def train_and_evaluate(params, criterion_class, criterion_params):
    model = Model(input_size, output_size, params['n_layers'], params['n_nodes'], params['activations'],
                  params['L1_penalty'], params['L2_penalty'], True, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'])
    criterion = criterion_class(**criterion_params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-8, verbose=True)
    early_stopping = EarlyStopping(monitor='val_mae', patience=50, min_delta=0.00001, verbose=True)

    for epoch in range(n_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Train Acc: {train_acc:.6f}, Val Acc: {val_acc:.6f}')
        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping at epoch:", epoch+1)
            break
    return val_loss, val_acc







