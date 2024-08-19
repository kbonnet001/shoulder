import os
import pandas as pd

class CSVBatchWriterTestHyperparams:
    def __init__(self, filename, batch_size=100):
        """ Initialize the CSVBatchWriterTestHyperparams with a file name, a list of degrees of freedom names,
        and an optional batch size.
        
        Args:
        - filename (str): The name - path of the CSV file to write data.
        - batch_size (int): The size of the batch for buffering data before writing to CSV.
        """
        self.filename = filename
        self.batch_size = batch_size
        self.buffer = [] # Initialize an empty list to act as a buffer for batch writing
        
        # Check if file exists, if not create it with initial structure
        if not os.path.exists(filename):
            data = {
                "num_try": [],
                "mode": [], 
                "val_loss": [], 
                "test_acc": [], 
                "test_error": [],
                "test_abs_error": [],
                "execution_time_train": [], 
                "execution_time_load_saved_model": [], 
                "execution_time_use_saved_model": [], 
                "batch_size": [], 
                "n_nodes": [], 
                "activations": [], 
                "L1_penalty": [], 
                "L2_penalty": [], 
                "learning_rate": [], 
                "dropout_prob": [], 
                "use_batch_norm": [], 
                "num_epoch_used": [], 
                "criterion_name": [], 
                "criterion_params": [], 
                 }
            # Create a DataFrame from the dictionary and write it to a CSV file
            pd.DataFrame(data).to_csv(filename, index=False)

    def add_line(self, num_try, val_loss, test_acc, test_error, test_abs_error, train_timer, mean_model_load_timer, 
                 mean_model_timer, try_hyperparams, mode, epoch, criterion_name, criterion_params):
        """ Add a new line of data to the buffer, containing information about a model training attempt. 
        If the buffer reaches the batch size, write it to the CSV file.

        Args:
        - num_try (int): The attempt number.
        - val_loss (float): Validation loss value.
        - test_acc (float): Validation accuracy value.
        - test_error (float): Validation error value.
        - test_abs_error (float): Validation absolute error value.
        - train_timer (float): Time taken for training execution.
        - mean_model_load_timer (float): Average time to load the saved model.
        - mean_model_timer (float): Average time to use the saved model.
        - try_hyperparams (ModelHyperparameters): Object containing hyperparameters used in the attempt.
        - mode (str Mode): Mode of the experiment or training run.
        - epoch (int): Number of epochs used in training.
        - criterion_name (str): Name of the criterion used for training.
        - criterion_params (dict): Parameters of the criterion used.
        """
        
        # Create a new line with the provided data
        new_line = {
            "num_try": num_try,
            "mode": mode, 
            "val_loss": val_loss, 
            "test_acc": test_acc, 
            "test_error": test_error,
            "test_abs_error": test_abs_error,
            "execution_time_train": train_timer, 
            "execution_time_load_saved_model": mean_model_load_timer, 
            "execution_time_use_saved_model": mean_model_timer, 
            "batch_size": try_hyperparams.batch_size, 
            "n_nodes": try_hyperparams.n_nodes, 
            "activations": try_hyperparams.activations, 
            "L1_penalty": try_hyperparams.L1_penalty, 
            "L2_penalty": try_hyperparams.L2_penalty, 
            "learning_rate": try_hyperparams.learning_rate, 
            "dropout_prob": try_hyperparams.dropout_prob, 
            "use_batch_norm": try_hyperparams.use_batch_norm, 
            "num_epoch_used": epoch, 
            "criterion_name": criterion_name, 
            "criterion_params": criterion_params, 
        }
        
        # Add the new line to the buffer
        self.buffer.append(new_line)
        
        # If buffer reaches the batch size, write to the file
        if len(self.buffer) >= self.batch_size:
            self._flush()
    
    def add_line_full(self, num_try, val_loss, val_acc, val_error, val_abs_error, train_timer, mean_model_load_timer, 
                 mean_model_timer, batch_size, n_nodes, activations, L1_penalty, L2_penalty, learning_rate, 
                 dropout_prob, use_batch_norm, mode, epoch, criterion_name, criterion_params):
        
        """ Add a new line of data to the buffer, containing detailed information about a model training attempt.
        If the buffer reaches the batch size, write it to the CSV file.

        Args:
        - num_try (int): The attempt number.
        - val_loss (float): Validation loss value.
        - test_acc (float): Validation accuracy value.
        - test_error (float): Validation error value.
        - test_abs_error (float): Validation absolute error value.
        - train_timer (float): Time taken for training execution.
        - mean_model_load_timer (float): Average time to load the saved model.
        - mean_model_timer (float): Average time to use the saved model.
        - batch_size (int): Batch size used in the training.
        - n_nodes (int): Number of nodes in the neural network.
        - activations (list): List of activation functions used.
        - L1_penalty (float): L1 regularization penalty value.
        - L2_penalty (float): L2 regularization penalty value.
        - learning_rate (float): Learning rate used during training.
        - dropout_prob (float): Dropout probability used in the network.
        - use_batch_norm (bool): Indicates whether batch normalization is used.
        - mode (str Mode): Mode of the experiment or training run.
        - epoch (int): Number of epochs used in training.
        - criterion_name (str): Name of the criterion used for training.
        - criterion_params (dict): Parameters of the criterion used.
        """
        # Create a new line with the provided data
        new_line = {
            "num_try": num_try,
            "mode": mode, 
            "val_loss": val_loss, 
            "test_acc": val_acc, 
            "test_error": val_error,
            "test_abs_error": val_abs_error,
            "execution_time_train": train_timer, 
            "execution_time_load_saved_model": mean_model_load_timer, 
            "execution_time_use_saved_model": mean_model_timer, 
            "batch_size": batch_size, 
            "n_nodes": n_nodes, 
            "activations": activations, 
            "L1_penalty": L1_penalty, 
            "L2_penalty": L2_penalty, 
            "learning_rate": learning_rate, 
            "dropout_prob": dropout_prob, 
            "use_batch_norm": use_batch_norm, 
            "num_epoch_used": epoch, 
            "criterion_name": criterion_name, 
            "criterion_params": criterion_params, 
        }
        
        # Add the new line to the buffer
        self.buffer.append(new_line)
        
        # If buffer reaches the batch size, write to the file
        if len(self.buffer) >= self.batch_size:
            self._flush()

    def _flush(self):
        """  Write the buffered data to the CSV file. If the file already exists, append the buffered data
        to the existing data; otherwise, create a new file with the buffered data.
        """
        if not self.buffer:
            return

        # Convert the buffer into a DataFrame
        buffer_df = pd.DataFrame(self.buffer)

        # Check if the CSV file already exists
        if os.path.exists(self.filename):
            # Read existing data
            df = pd.read_csv(self.filename)
            # Concatenate existing data with buffer
            df = pd.concat([df, buffer_df], ignore_index=True)
        else:
            # If file doesn't exist, use the buffer data
            df = buffer_df

        # Write the updated DataFrame to the CSV file
        df.to_csv(self.filename, index=False)

        # Clear the buffer
        self.buffer = []

    def del_lines(self, n):
        """ Delete excess lines from the CSV file, keeping only the first 'n' lines.

        Args:
        - n (int): The number of lines to retain in the CSV file.
        """
        # Read the data from the CSV file
        df = pd.read_csv(self.filename)
        current_lines = df.shape[0]

        # If the current number of lines is greater than 'n', delete the excess lines
        if current_lines > n:
            df = df.head(n) # Retain only the first 'n' lines

            # Write the updated DataFrame back to the CSV file
            df.to_csv(self.filename, index=False)
                
    def get_num_line(self) : 
        """ Get the number of lines currently in the CSV file.

        Returns:
        int: The number of lines in the CSV file. Returns 0 if the file does not exist.
        """
        # Check if the CSV file exists
        if os.path.exists(self.filename):
            df = pd.read_csv(self.filename)
            return len(df)
        else:
            # If the file does not exist, return 0
            return 0
        
    def close(self):
        """
        Flush any remaining lines in the buffer to the CSV file when closing.
        """
        # Write any remaining buffered data to the CSV file
        self._flush()

