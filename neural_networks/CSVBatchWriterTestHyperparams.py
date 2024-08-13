import os
import pandas as pd

class CSVBatchWriterTestHyperparams:
    def __init__(self, filename, batch_size=100):
        self.filename = filename
        self.batch_size = batch_size
        self.buffer = []
        
        # Check if file exists, if not create it with initial structure
        if not os.path.exists(filename):
            data = {
                "num_try": [],
                "mode": [], 
                "val_loss": [], 
                "val_acc": [], 
                "val_error": [],
                "val_abs_error": [],
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
            pd.DataFrame(data).to_csv(filename, index=False)

    def add_line(self, num_try, val_loss, val_acc, val_error, val_abs_error, train_timer, mean_model_load_timer, 
                 mean_model_timer, try_hyperparams, mode, epoch, criterion_name, criterion_params):
        # Create a new line with the provided data
        new_line = {
            "num_try": num_try,
            "mode": mode, 
            "val_loss": val_loss, 
            "val_acc": val_acc, 
            "val_error": val_error,
            "val_abs_error": val_abs_error,
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
        # Create a new line with the provided data
        new_line = {
            "num_try": num_try,
            "mode": mode, 
            "val_loss": val_loss, 
            "val_acc": val_acc, 
            "val_error": val_error,
            "val_abs_error": val_abs_error,
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
        # Lire le fichier CSV
        df = pd.read_csv(self.filename)
        current_lines = df.shape[0]

        # Si le nombre de lignes actuel est supérieur à n, supprimer l'excédent
        if current_lines > n:
            df = df.head(n)  # Conserver seulement les n premières lignes

            # Écrire le DataFrame mis à jour dans le fichier CSV
            df.to_csv(self.filename, index=False)

    def get_num_line(self) : 
        if os.path.exists(self.filename):
            df = pd.read_csv(self.filename)
            return len(df)
        else:
            return 0
        
    def close(self):
        # Flush any remaining lines in the buffer when closing
        self._flush()
