import torch
from neural_networks.Model import Model
import torch.optim as optim
from neural_networks.EarlyStopping import EarlyStopping
from neural_networks.save_model import *
from neural_networks.plot_visualisation import *
import math

def convert_tensor_to_numpy(tensor) : 
    """
    Convert a PyTorch tensor of predictions to a 1D NumPy array.

    Args:
    - predictions_tensor: A PyTorch tensor containing predictions.

    Returns:
    - A 1D NumPy array of predictions.
    """
    # Concatenate predictions and targets
    tensor = torch.cat(tensor)

    # Squeeze the dimensions of size 1
    tensor = tensor.squeeze(1)  # Remove dimensions of size 1

    # Convert to 1D NumPy arrays
    numpy_array = tensor.detach().cpu().numpy()
    
    return numpy_array

def train(model, train_loader, optimizer, criterion, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Train the model on the training dataset.

    Args:
    - model (nn.Module): The neural network model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset, which provides batches of data.
    - optimizer (torch.optim.Optimizer): The optimization algorithm to update the model's weights.
    - criterion (torch.nn.Module): The loss function to minimize during training.
    - device (torch.device, optional): The device on which to perform computations (CPU or CUDA). 
        Default is CUDA if available.

    Returns:
    - epoch_loss (float): The average loss over the training dataset for the current epoch.
    - epoch_acc (float): The accuracy of the model on the training dataset for the current epoch.
    - epoch_pourcentage_error (float): The percentage error of predictions compared to targets.
    - abs_epoch_pourcentage_error (float): The absolute percentage error of predictions compared to targets.
    """

    model.train()  # Train mode
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() # Gradients initialization
        outputs = model(inputs) # Model prediction

        # Ensure targets have the same dimensions as outputs
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
    
    # Convert lists of tensors to NumPy arrays for further analysis
    all_predictions_np = convert_tensor_to_numpy(all_predictions)
    all_targets_np = convert_tensor_to_numpy(all_targets)

    # Calculation of mean distance and error %
    epoch_acc = mean_distance(all_predictions_np, all_targets_np)
    epoch_pourcentage_error, abs_epoch_pourcentage_error = compute_pourcentage_error(all_predictions_np, all_targets_np)

    return epoch_loss, epoch_acc, epoch_pourcentage_error, abs_epoch_pourcentage_error

def evaluate(model, data_loader, criterion, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Evaluate the model on the validation or test dataset.

    Args:
    - model (nn.Module): The neural network model to be evaluated.
    - data_loader (DataLoader): DataLoader for the validation or test dataset.
    - criterion (torch.nn.Module): The loss function to calculate the error between predictions and true values.
    - device (torch.device, optional): The device on which to perform computations (CPU or CUDA). Default is CUDA if available.

    Returns:
    - epoch_loss (float): The average loss over the training dataset for the current epoch.
    - epoch_acc (float): The accuracy of the model on the training dataset for the current epoch.
    - epoch_pourcentage_error (float): The percentage error of predictions compared to targets.
    - abs_epoch_pourcentage_error (float): The absolute percentage error of predictions compared to targets.
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

    # Convert lists of tensors to NumPy arrays for further analysis
    all_predictions_np = convert_tensor_to_numpy(all_predictions)
    all_targets_np = convert_tensor_to_numpy(all_targets)

    # Calculation of mean distance and error %
    epoch_acc = mean_distance(all_predictions_np, all_targets_np)
    epoch_pourcentage_error, abs_epoch_pourcentage_error = compute_pourcentage_error(all_predictions_np, all_targets_np)

    return epoch_loss, epoch_acc, epoch_pourcentage_error, abs_epoch_pourcentage_error

def train_model_supervised_learning(train_loader, val_loader, test_loader, input_size, output_size, Hyperparams, 
                                    file_path, plot = False, save = False, show_plot = False) : 
    """Train and evaluate a model.
    
    Args:
    - train_loader (DataLoader): DataLoader for the training dataset (80% of 80%).
    - val_loader (DataLoader): DataLoader for the validation dataset (20% of 80%).
    - test_loader (DataLoader): DataLoader for the test dataset (20%).
    - input_size (int): Size of input features X.
    - output_size (int): Size of output y 
    - Hyperparams (ModelHyperparameters): Hyperparameters for the model.
    - file_path (string): Path for saving the model.
    - plot (bool, default=False): Whether to show and save plots.
    - save (bool, default=False): Whether to save the model.
    
    Returns:
    - model (nn.Module): Trained model.
    - val_loss (float): Validation loss.
    - val_acc (float): Validation accuracy (mean distance).
    - val_error (float): Validation error.
    - val_abs_error (float): Validation absolute error.
    - epoch (int): Number of epochs completed.
    """
    # Initialize the model with given hyperparameters
    model = Model(input_size, output_size, Hyperparams.n_nodes, Hyperparams.activations, 
                  Hyperparams.L1_penalty, Hyperparams.L2_penalty, Hyperparams.use_batch_norm, Hyperparams.dropout_prob)
    
    # Set up optimizer and learning rate scheduler
    Hyperparams.compute_optimiser(model) # compute the optimiser
    min_lr = 1e-8 # min lr could be attend with scheduler
    patience_scheduler = 20 
    patience_early_stopping = 40 # Choose a early-stopping patience = 2 * scheduler patience
        
    # More details about scheduler in documentation
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(Hyperparams.optimizer, mode='min', factor=0.1, 
                                                     patience=patience_scheduler, min_lr=min_lr)
    # Initialization of EarlyStopping
    # More informations about Early stopping in doccumentation
    early_stopping = EarlyStopping(monitor='val_mae', patience=patience_early_stopping, min_delta=1e-9, verbose=True)
    
    # Prepare for plot
    if plot : 
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        train_errors = []
        val_errors = []
        train_abs_errors = []
        val_abs_errors = []

    for epoch in range(Hyperparams.num_epochs):
        train_loss, train_acc, train_error, train_abs_error = train(model, train_loader, Hyperparams.optimizer, 
                                                                    Hyperparams.criterion)
        val_loss, val_acc, val_error, val_abs_error = evaluate(model, val_loader, Hyperparams.criterion)
        
        # Check for NaN values in accuracy
        # Sometimes, acc(s) could be Nan :(
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
        
        #  if patience_scheduler, adjust/reduce learning rate 
        scheduler.step(val_loss)

        # if patience_early_stopping, stop trainning to avoid overfitting
        early_stopping(val_loss)
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






