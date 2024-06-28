import matplotlib as plt
import matplotlib.pyplot as plt
import torch
from neural_networks.data_tranning import mean_distance

def plot_datas_distribution(X_tensor, y_tensor):
    """To visualise tensors distribution

    INPUT : 
    - X_tensor : X tensor with all features (columns except the last one)
    - y_tensor : y tensor with the target values (last column) """
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10)) 
    
    for i in range(4):
        row = i // 3  
        col = i % 3   
        axs[row, col].hist(X_tensor[:, i], bins=20, alpha=0.5)
        axs[row, col].set_xlabel('Value')
        axs[row, col].set_ylabel('Frequency')
        axs[row, col].set_title(f'Distribution of q{i+1}')

    axs[1, 2].hist(y_tensor, bins=20, alpha=0.5)  
    axs[1, 2].set_xlabel('Value')
    axs[1, 2].set_ylabel('Frequency')
    axs[1, 2].set_title('Distribution of muscle length')

    plt.tight_layout()  
    plt.show()

def plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs):
    """Plot loss and accuracy (train and validation)

    INPUT :
    - train_losses :
    - val_losses
    - train_accs
    - val_accs """
    # Créer une figure avec deux sous-graphiques
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Tracer les courbes de pertes
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Train and Validation Loss over Epochs')
    axs[0].legend()

    # Tracer les courbes d'accuracy
    axs[1].plot(train_accs, label='Train Accuracy')
    axs[1].plot(val_accs, label='Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Train and Validation Accuracy over Epochs')
    axs[1].legend()

    # Afficher la figure
    plt.tight_layout()
    plt.show()
    
def get_predictions_and_targets(model, data_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    """Get predictions and targets from a model and data loader.

    INPUT:
    - model: The trained PyTorch model to be evaluated.
    - data_loader: DataLoader containing the dataset to evaluate.
    - device: The device to run the model on (default is CUDA if available, otherwise CPU).

    OUTPUT:
    - predictions: A list of predictions made by the model.
    - targets: A list of true target values from the dataset.
    """
    
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return predictions, targets

    
def plot_predictions_and_targets(model, loader, string_loader, num) :
    
    """Plot the true values and predicted values for a given model and data loader.
    INPUT:
    - model: The trained PyTorch model to be evaluated.
    - loader: DataLoader containing the dataset to evaluate.
    - num: The number of samples to plot for comparison.

    OUTPUT:
    - None: The function generates a plot showing the true values and predicted values.
    """

    # # Obtain predictions and true values
    # predictions, targets = get_predictions_and_targets(model, loader)
    
    # # Convertir les listes en numpy.ndarray
    # predictions_np = np.array(predictions)
    # targets_np = np.array(targets)

    # # Convertir les numpy.ndarray en torch.Tensor
    # predictions_tensor = torch.tensor(predictions_np)
    # targets_tensor = torch.tensor(targets_np)

    # # Calculer la mean_distance
    # acc = mean_distance(predictions_tensor, targets_tensor)
    
    # # acc = mean_distance(torch.tensor(predictions), torch.tensor(targets))
    # print("acc = ", acc)


    # Obtenir les prédictions et les valeurs réelles pour l'ensemble de test
    predictions, targets = get_predictions_and_targets(model, loader)

    acc = mean_distance(torch.tensor(predictions), torch.tensor(targets))
    print("acc = ", acc)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(targets[:num], label='True values', marker='o')
    plt.plot(predictions[:num], label='Predictions', marker='o',linestyle='--')
    plt.xlabel('Échantillons')
    plt.ylabel('Muscle length')
    plt.title(f"Predictions and targets - {string_loader}")
    plt.legend()
    plt.show()
