import torch

class ModelHyperparameters:
    def __init__(self, model_name, mode, batch_size, n_nodes, activations, activation_names, L1_penalty, L2_penalty, 
                 learning_rate, num_epochs, criterion, dropout_prob, use_batch_norm):
        
        self.model_name = model_name
        self.mode = mode
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
        self.loss = 0.0
        self.mean_distance = 0.0
        
    def save_results_parameters(self, loss, mean_distance) : 
        self.loss = loss
        self.mean_distance = mean_distance
        
    def compute_optimiser(self, model) :
        self.optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
    
    def add_criterion(self, criterion_params) : 
        self.criterion = criterion_params

    def __str__(self):
        return (f"ModelConfig : \n- model_name={self.model_name},\n- mode={self.mode},\n- batch_size={self.batch_size},\n"
                f"- n_nodes={self.n_nodes},\n- activations={self.activations},\n"
                f"- L1_penalty={self.L1_penalty},\n- L2_penalty={self.L2_penalty},\n- learning_rate={self.learning_rate},\n"
                f"- num_epochs={self.num_epochs},\n- optimizer={self.optimizer},\n- criterion={self.criterion},\n" 
                f"- p_dropout={self.dropout_prob}, \n- use_batch_norm={self.use_batch_norm},\n- loss={self.loss},\n"
                f"- mean_distance={self.mean_distance}")
                
                
                
                
                
                
                
               
