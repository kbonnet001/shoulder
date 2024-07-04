import torch
import matplotlib.pyplot as plt
from neural_networks.Model import Model
import torch.optim as optim
from neural_networks.EarlyStopping import EarlyStopping
from neural_networks.save_model import *
from neural_networks.plot_visualisation import *
from neural_networks.file_directory_operations import create_and_save_plot

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

def train_model_supervised_learning(train_loader, val_loader, test_loader, input_size, output_size, Hyperparams, 
                                    file_path, plot = False, save = False) : 
    """Train and evaluate a model
    
    INPUTS : 
    - train_loader : DataLoader, data trainning (80% of 80%)
    - val_loader : DataLoader, data validation (20% of 80%)
    - test_loader : DataLoader, data testing (20%)
    - input_size : int, size of input X
    - output_size : int, size of output y (WARNING, always 1)
    - Hyperparams : (ModelHyperparameters) all hyperparameters choosen by user
    - file_path : string, path for saving model
    - plot : (default False) bool, True to show and save plots
    - save : (default False) bool, True to save the model
    
    OUTPUTS : 
    - val_loss : float, loss validation
    - val_acc : float, accuracy (mean distance) validation"""
    
    model = Model(input_size, output_size, Hyperparams.n_layers, Hyperparams.n_nodes, Hyperparams.activations, 
                  Hyperparams.L1_penalty, Hyperparams.L2_penalty, Hyperparams.use_batch_norm, Hyperparams.dropout_prob)
    
    Hyperparams.compute_optimiser(model)
    
    if plot : 
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

    # Initialization of ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(Hyperparams.optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-8, verbose=True)
    # Initialization of EarlyStopping
    early_stopping = EarlyStopping(monitor='val_mae', patience=50, min_delta=0.00001, verbose=True)

    for epoch in range(Hyperparams.num_epochs):
        train_loss, train_acc = train(model, train_loader, Hyperparams.optimizer, Hyperparams.criterion)
        val_loss, val_acc = evaluate(model, val_loader, Hyperparams.criterion)
        
        if plot : 
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

        print(f'Epoch [{epoch+1}/{Hyperparams.num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Train Acc: {train_acc:.6f}, Val Acc: {val_acc:.6f}')

        # Réduire le taux d'apprentissage si nécessaire
        scheduler.step(val_loss)

        # Vérifier l'arrêt précoce
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping at epoch:", epoch+1)
            break

    # Évaluation du modèle sur l'ensemble de test
    test_loss, test_acc = evaluate(model, test_loader, Hyperparams.criterion)
    print(f'Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}')
    
    if plot : 
        plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs)
        create_and_save_plot(Hyperparams.model_name, "plot_loss_and_accuracy")
        
        plot_predictions_and_targets(model, train_loader, "Train loader", 100)
        create_and_save_plot(Hyperparams.model_name, "plot_predictions_and_targets_train_loader")
        plot_predictions_and_targets(model, val_loader, "Validation loader", 100)
        create_and_save_plot(Hyperparams.model_name, "plot_predictions_and_targets_val_loader")
        plot_predictions_and_targets(model, test_loader, "Test loader", 100)
        create_and_save_plot(Hyperparams.model_name, "plot_predictions_and_targets_test_loader")
    
    # Save model
    if save : 
        save_model(model, input_size, output_size, Hyperparams, file_path)
    
    return model, val_loss, val_acc







