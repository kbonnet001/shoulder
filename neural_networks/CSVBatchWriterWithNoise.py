import pandas as pd
import numpy as np
import os

class CSVBatchWriterWithNoise:
    def __init__(self, filename, q_ranges_names_with_dofs, batch_size=1000, noise_std_dev=0.01):
        self.filename = filename
        self.q_ranges_names_with_dofs = q_ranges_names_with_dofs
        self.batch_size = batch_size
        self.noise_std_dev = noise_std_dev
        self.buffer = []
        
        # Check if file exists, if not create it with initial structure
        if not os.path.exists(filename):
            data = {
                "muscle_selected": [],
                **{f"q_{self.q_ranges_names_with_dofs[k]}": [] for k in range(len(self.q_ranges_names_with_dofs))},
                **{f"qdot_{self.q_ranges_names_with_dofs[k]}": [] for k in range(len(self.q_ranges_names_with_dofs))},
                **{f"alpha_{self.q_ranges_names_with_dofs[k]}": [] for k in range(len(self.q_ranges_names_with_dofs))},
                "origin_muscle_x": [],
                "origin_muscle_y": [],
                "origin_muscle_z": [],
                "insertion_muscle_x": [],
                "insertion_muscle_y": [],
                "insertion_muscle_z": [],
                "segment_length": [],
                **{f"dlmt_dq_{self.q_ranges_names_with_dofs[k]}": [] for k in range(len(self.q_ranges_names_with_dofs))},
                "muscle_force": [],
                "torque": []
                 }
            pd.DataFrame(data).to_csv(filename, index=False)
        
    
    def add_line(self, new_line):
        
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
    
    def augment_data_with_noise_batch(self, dataset_size_noise):
        # Warning, len(df) must be a multiple of self.batch_size to add the correct num of row
        df = pd.read_csv(f"{self.filename.replace("_with_noise", "")}")

        num_rows_to_add = dataset_size_noise - len(df) 
        
        if num_rows_to_add > 0 :
            num_lines_to_add_per_batch = int(self.batch_size /len(df) * num_rows_to_add)
            num_chunks = len(df) // self.batch_size + 1
            
            for i in range(num_chunks - 1):
                chunk = df[i*self.batch_size : (i+1)*self.batch_size]
                self.add_noise_to_batch(chunk, num_lines_to_add_per_batch)
            
            # Flush remaining data in buffer
            self._flush()

    def add_noise_to_batch(self, df_batch, num_rows_to_add):
        augmented_data = []
        
        for _ in range(num_rows_to_add):
            # Sélectionner une ligne aléatoire à bruiter, à l'exception de la première colonne
            row = df_batch.iloc[np.random.randint(len(df_batch)), 1:].values
            
            # Ajouter du bruit aux colonnes restantes
            noise = np.random.normal(0, self.noise_std_dev, row.shape)
            noisy_row = row + noise
            
            # Créer une liste avec la première colonne et les colonnes bruitées
            noisy_row_list = [df_batch.iloc[0, 0]] + noisy_row.tolist()
            
            # Ajouter la ligne bruitée à la liste augmentée
            augmented_data.append(noisy_row_list)

        for l_idx in augmented_data : 
            self.add_line(l_idx)
        
