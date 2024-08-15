import torch
from neural_networks.Model import Model
import torch.optim as optim
from neural_networks.EarlyStopping import EarlyStopping
from neural_networks.save_model import *
from neural_networks.plot_visualisation import *
import math

def train(model, train_loader, optimizer, criterion, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Train the model on the training dataset.

    Args :
    - model (nn.Module): The neural network model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset, which provides batches of data.
    - optimizer (torch.optim.Optimizer): The optimization algorithm to update the model's weights.
    - criterion (torch.nn.Module): The loss function to minimize during training.
    - device (torch.device, optional): The device on which to perform computations (CPU or CUDA). Default is CUDA if available.

    Returns :
    - epoch_loss (float): The average loss over the training dataset for the current epoch.
    - epoch_acc (float): The accuracy of the model on the training dataset for the current epoch.
    """

    model.train()  # Train mode
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() # Gradients initialization
        outputs = model(inputs) # Model prediction

        if len(targets.size()) == 1 : # security when y is 1
            targets = targets.unsqueeze(1)
        loss = criterion(outputs, targets) # Compute loss
        loss.backward() # Backpropagation
        optimizer.step() # Updating weights

        # Stats for plot
        running_loss += loss.item() * inputs.size(0)
        all_predictions.append(outputs)
        all_targets.append(targets)

    # Calculation of average loss
    epoch_loss = running_loss / len(train_loader.dataset)

    # Calculation of mean distance
    
    # outputs = outputs.squeeze(1) # Remove dimensions of size 1
    # labels = labels.squeeze(1) # Remove dimensions of size 1
    # # Convert tensors to numpy arrays and extend the lists
    # predictions.extend(outputs.cpu().numpy())
    
    # Concatenate predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    # Squeeze the dimensions of size 1
    all_predictions = all_predictions.squeeze(1)  # Remove dimensions of size 1
    all_targets = all_targets.squeeze(1)          # Remove dimensions of size 1

    # Convert to 1D NumPy arrays
    all_predictions_np = all_predictions.detach().cpu().numpy()
    all_targets_np = all_targets.detach().cpu().numpy()
    
    epoch_acc = mean_distance(all_predictions_np, all_targets_np)
    epoch_pourcentage_error, abs_epoch_pourcentage_error = compute_pourcentage_error(all_predictions_np, all_targets_np)

    return epoch_loss, epoch_acc, epoch_pourcentage_error, abs_epoch_pourcentage_error

def evaluate(model, data_loader, criterion, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Evaluate the model on the validation or test dataset.

    Args :
    - model (nn.Module): The neural network model to be evaluated.
    - data_loader (DataLoader): DataLoader for the validation or test dataset, which provides batches of data.
    - criterion (torch.nn.Module): The loss function to calculate the error between the model predictions and true values.
    - device (torch.device, optional): The device on which to perform computations (CPU or CUDA). Default is CUDA if available.

    Returns :
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

            if len(targets.size()) == 1 : # security when y is 1
                targets = targets.unsqueeze(1)
            loss = criterion(outputs, targets)  # Compute loss

            # Stats for plot
            running_loss += loss.item() * inputs.size(0)
            all_predictions.append(outputs)
            all_targets.append(targets)

    # Calculation of average loss
    epoch_loss = running_loss / len(data_loader.dataset)

    # Concatenate predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    # Squeeze the dimensions of size 1
    all_predictions = all_predictions.squeeze(1)  # Remove dimensions of size 1
    all_targets = all_targets.squeeze(1)          # Remove dimensions of size 1

    # Convert to 1D NumPy arrays
    all_predictions_np = all_predictions.detach().cpu().numpy()
    all_targets_np = all_targets.detach().cpu().numpy()
    
    epoch_acc = mean_distance(all_predictions_np, all_targets_np)
    epoch_pourcentage_error, abs_epoch_pourcentage_error = compute_pourcentage_error(all_predictions_np, all_targets_np)

    return epoch_loss, epoch_acc, epoch_pourcentage_error, abs_epoch_pourcentage_error

def train_model_supervised_learning(train_loader, val_loader, test_loader, input_size, output_size, Hyperparams, 
                                    file_path, plot = False, save = False, show_plot = False) : 
    """Train and evaluate a model
    
    Args : 
    - train_loader : DataLoader, data trainning (80% of 80%)
    - val_loader : DataLoader, data validation (20% of 80%)
    - test_loader : DataLoader, data testing (20%)
    - input_size : int, size of input X
    - output_size : int, size of output y (WARNING, always 1)
    - Hyperparams : (ModelHyperparameters) all hyperparameters choosen by user
    - file_path : string, path for saving model
    - plot : (default False) bool, True to show and save plots
    - save : (default False) bool, True to save the model
    
    Returns : 
    - val_loss : float, loss validation
    - val_acc : float, accuracy (mean distance) validation"""
    
    model = Model(input_size, output_size, Hyperparams.n_nodes, Hyperparams.activations, 
                  Hyperparams.L1_penalty, Hyperparams.L2_penalty, Hyperparams.use_batch_norm, Hyperparams.dropout_prob)
    
    Hyperparams.compute_optimiser(model)
    
    if plot : 
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        train_errors = []
        val_errors = []
        train_abs_errors = []
        val_abs_errors = []
        
    # Initialization of ReduceLROnPlateau
    min_lr=1e-8
    patience_scheduler=20
    early_stop_scheduler = 0
    
    # choose a early-stopping patience = 2 * scheduler patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(Hyperparams.optimizer, mode='min', factor=0.1, patience=patience_scheduler, min_lr=min_lr)
    # Initialization of EarlyStopping
    early_stopping = EarlyStopping(monitor='val_mae', patience=40, min_delta=1e-9, verbose=True)

    for epoch in range(Hyperparams.num_epochs):
        train_loss, train_acc, train_error, train_abs_error = train(model, train_loader, Hyperparams.optimizer, Hyperparams.criterion)
        val_loss, val_acc, val_error, val_abs_error = evaluate(model, val_loader, Hyperparams.criterion)
        
        # Security
        # Sometime, acc(s) could be Nan :(
        # Check your activation function and try an other !
        if math.isnan(train_acc) or math.isnan(val_acc):
            return model, float('inf'), float('inf')
        
        if plot : 
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_errors.append(train_error)
            val_errors.append(val_error)
            train_abs_errors.append(train_abs_error)
            val_abs_errors.append(val_abs_error)

        print(f'Epoch [{epoch+1}/{Hyperparams.num_epochs}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f},',\
              f"Train Acc: {train_acc:.6f}, Val Acc: {val_acc:.6f}, lr = {scheduler.get_last_lr()}")
        print("early_stop_scheduler = ", early_stop_scheduler)
        
        # Réduire le taux d'apprentissage si nécessaire
        scheduler.step(val_loss)

        # Vérifier l'arrêt précoce
        early_stopping(val_loss)
        # if early_stopping.early_stop : 
            # early_stop_scheduler += 1
            
        if early_stopping.early_stop :
            print("Early stopping at epoch:", epoch+1)
            break

    # Évaluation du modèle sur l'ensemble de test
    test_loss, test_acc, test_error, test_abs_error= evaluate(model, test_loader, Hyperparams.criterion)
    print(f'Test Loss: {test_loss:.8f}, Test Acc: {test_acc:.8f}, Test Error: {test_error:.8f}, \
        Test Abs Error: {test_abs_error:.8f}')
    
    if plot : 
        plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs, train_errors, val_errors, train_abs_errors, 
                               val_abs_errors, file_path, show_plot)
        
    # Save model
    if save : 
        save_model(model, input_size, output_size, Hyperparams, f"{file_path}")
    
    return model, val_loss, val_acc, val_error, val_abs_error, epoch







