import torch

class ModelTryHyperparameters:
    def __init__(self, model_name, batch_size, n_nodes, activations, activation_names, L1_penalty, L2_penalty, 
                 learning_rate, num_epochs, criterion, dropout_prob, use_batch_norm):
        """ Initialize the hyperparameters for a machine learning model with find_best_hyperparameters.
        Don't use it to try various hyperparameters; use it for only one model.
        Please see examples below to better understand.
        To understand how to choose hyperparameters, please refer to the documentation.

        Args:
        - model_name (str): Name of the model.
        - batch_size (int): Batch size used for training.
        - n_nodes (list of lists of int): Number of nodes in the neural network.
        - activations (list of lists): List of activation functions used in the model.
        - activation_names (list of lists): Names of the activation functions used.
        - L1_penalty (list of float): L1 regularization penalty value.
        - L2_penalty (list of float): L2 regularization penalty value.
        - learning_rate (list of float): Learning rate for the optimizer.
        - num_epochs (int): Number of epochs for training.
        - criterion (list of torch.nn.Module): Loss function used for training.
        - dropout_prob (list of float): Dropout probability used in the network.
        - use_batch_norm (bool): Flag indicating if batch normalization is used.
        
        Example : 
        model_name = "example"
        batch_size = 64  # Conventionally, use multiples of 2, such as 32, 64, 128, etc.
        n_nodes = [[32, 32], [64, 64]] # or [[32, 32]]
        activations = [[nn.GELU(), nn.GELU()], [nn.ReLU(), nn.ReLU()]] # or [[nn.GELU(), nn.GELU()]]
        activation_names = [["GELU", "GELU"], ["ReLU", "ReLU"]] # or [["GELU", "GELU"]]
        L1_penalty = [0.01, 0.001] # or [0.01]
        L2_penalty = [0.01, 0.001] # or [0.01]
        learning_rate = [1e-2, 1e-3] # or [1e-2]
        num_epochs = 1000
        criterion = [
            (LogCoshLoss, {'factor': [1.0, 1.8]}),
            (ModifiedHuberLoss, {'delta': [0.2, 0.5, 1.0], 'factor': [0.5, 1.0, 1.5]}),
            (ExponentialLoss, {'alpha': [0.5, 1.0]}),
            (nn.MSELoss, {})
        dropout_prob = [0.0, 0.2] # or [0.2]
        use_batch_norm = True
        """
        
        # Verify that n_nodes, activations, and activation_names are lists of lists
        if not (self._is_list_of_lists(n_nodes) and self._is_list_of_lists(activations) 
                and self._is_list_of_lists(activation_names)):
            raise TypeError("n_nodes, activations, and activation_names must be lists of lists.")

        # Ensure that dropout_prob is a list of floats
        if not (isinstance(L1_penalty, list) and isinstance(L2_penalty, list) and isinstance(learning_rate, list) 
                and isinstance(dropout_prob, list)):
            raise TypeError(f"'L1_penalty', 'L2_penalty', 'learning_rate' and 'dropout_prob' must be a list.")
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.activations = activations
        self.activation_names = activation_names
        self.L1_penalty = L1_penalty
        self.L2_penalty = L2_penalty
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.dropout_prob = dropout_prob
        self.use_batch_norm = use_batch_norm
        self.optimizer = 0.0
        
    def compute_optimiser(self, model) :
        """ Initialize the optimizer for the model using the Adam optimization algorithm.

        Args:
        - model (torch.nn.Module): The model for which the optimizer is to be computed.
        """
        self.optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
    
    def add_criterion(self, criterion_params) : 
        """ Set the loss criterion for the model.

        Args:
        - criterion_params (torch.nn.Module): The loss function to be used for training.
        """
        self.criterion = criterion_params
    
    def _is_list_of_lists(self, obj):
        """ Helper function to check if an object is a list of lists """
        return (isinstance(obj, list) and all(isinstance(sublist, list) for sublist in obj))


    def __str__(self):
        return (f"ModelConfig : \n- model_name={self.model_name},\n- batch_size={self.batch_size},\n"
                f"- n_nodes={self.n_nodes},\n- activations={self.activations},\n"
                f"- L1_penalty={self.L1_penalty},\n- L2_penalty={self.L2_penalty},\n- learning_rate={self.learning_rate},\n"
                f"- num_epochs={self.num_epochs},\n- optimizer={self.optimizer},\n- criterion={self.criterion},\n" 
                f"- p_dropout={self.dropout_prob}, \n- use_batch_norm={self.use_batch_norm}")
                
                
                
                
                
                
                
               
