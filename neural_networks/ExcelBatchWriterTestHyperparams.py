import os
import pandas as pd

class ExcelBatchWriterTestHyperparams:
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
            pd.DataFrame(data).to_excel(filename, index=False)

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
        
        # Lire les données existantes du fichier Excel
        df = pd.read_excel(self.filename)
        
        # Convertir le buffer en DataFrame
        buffer_df = pd.DataFrame(self.buffer)
        
        # Vérifier si le DataFrame n'est pas vide ou entièrement NaN
        if not buffer_df.empty and not buffer_df.isna().all().all():
            buffer_df = buffer_df.dropna(axis=1, how='all')
            # Concaténer le DataFrame existant avec le buffer
            df = pd.concat([df, buffer_df], ignore_index=True)
        
            # Écrire le DataFrame mis à jour dans le fichier Excel
            with pd.ExcelWriter(self.filename, engine='openpyxl', mode='w') as writer:
                df.to_excel(writer, index=False)
        
        # Vider le buffer
        self.buffer = []

    def del_lines(self, n):
        # Check current number of lines in the Excel file
        df = pd.read_excel(self.filename)
        current_lines = df.shape[0]
        
        # If there are more than n lines, delete the surplus
        if current_lines > n:
            df.drop(df.tail(current_lines - n).index, inplace=True)
            
            # Write the updated DataFrame back to the Excel file
            with pd.ExcelWriter(self.filename, engine='openpyxl', mode='w') as writer:
                df.to_excel(writer, index=False)

    def get_num_line(self) : 
        if os.path.exists(self.filename):
            df = pd.read_excel(self.filename)
            return len(df)
        else:
            return 0
        
    def close(self):
        # Flush any remaining lines in the buffer when closing
        self._flush()
