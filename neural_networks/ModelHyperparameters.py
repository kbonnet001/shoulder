import torch

class ModelHyperparameters:
    def __init__(self, model_name, batch_size, n_nodes, activations, activation_names, L1_penalty, L2_penalty, 
                 learning_rate, num_epochs, criterion, dropout_prob, use_batch_norm):
        """ Initialize the hyperparameters for a machine learning model.
        Don't use it to try various hyperparameters; use it for only one model.
        Please see examples below to better understand.
        To understand how to choose hyperparameters, please refer to the documentation.

        Args:
        - model_name (str): Name of the model.
        - batch_size (int): Batch size used for training.
        - n_nodes (list of int): Number of nodes in the neural network.
        - activations (list): List of activation functions used in the model.
        - activation_names (list): Names of the activation functions used.
        - L1_penalty (float): L1 regularization penalty value.
        - L2_penalty (float): L2 regularization penalty value.
        - learning_rate (float): Learning rate for the optimizer.
        - num_epochs (int): Number of epochs for training.
        - criterion (torch.nn.Module): Loss function used for training.
        - dropout_prob (float): Dropout probability used in the network.
        - use_batch_norm (bool): Flag indicating if batch normalization is used.
        
        Example : 
        model_name = "example"
        batch_size = 64  # Conventionally, use multiples of 2, such as 32, 64, 128, etc.
        n_nodes = [32, 32] # 2 layers
        activations = [nn.GELU(), nn.GELU()]
        activation_names = ["GELU", "GELU"]]
        L1_penalty = 0.01
        L2_penalty = 0.01
        learning_rate = 1e-2
        num_epochs = 1000
        criterion = ModifiedHuberLoss(delta=0.2, factor=1.0)
        dropout_prob = 0.2 # [|0.0, 0.8|]
        use_batch_norm = True
        """
        
        # Verify that activations, activation_names, and n_nodes have the same length
        if not (len(activations) == len(activation_names) == len(n_nodes)):
            raise ValueError("The lengths of 'activations', 'activation_names', and 'n_nodes' must be the same.")
        
        # Ensure that L1_penalty and L2_penalty are floats
        if not (isinstance(L1_penalty, float) and isinstance(L2_penalty, float) and isinstance(learning_rate, float)
                and isinstance(dropout_prob, float)) :
            raise TypeError("'L1_penalty', 'L2_penalty', 'learning_rate' and 'dropout_prob' must be of type float.")

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

    def __str__(self):
        return (f"ModelConfig : \n- model_name={self.model_name},\n- batch_size={self.batch_size},\n"
                f"- n_nodes={self.n_nodes},\n- activations={self.activations},\n"
                f"- L1_penalty={self.L1_penalty},\n- L2_penalty={self.L2_penalty},\n- learning_rate={self.learning_rate},\n"
                f"- num_epochs={self.num_epochs},\n- optimizer={self.optimizer},\n- criterion={self.criterion},\n" 
                f"- p_dropout={self.dropout_prob}, \n- use_batch_norm={self.use_batch_norm}")
                
                
                
                
                
                
                
               
