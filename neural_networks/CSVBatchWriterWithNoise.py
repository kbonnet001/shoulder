import pandas as pd
import numpy as np
import os

class CSVBatchWriterWithNoise:
    def __init__(self, filename, q_ranges_names_with_dofs, nb_muscles, nb_q, batch_size=1000, noise_std_dev=0.01):
        """ Initialize the CSVBatchWriter instance.

        Args:
        - filename (str): Name - path of the CSV file to write data to.
        - q_ranges_names_with_dofs (list): List of names for the degrees of freedom.
        - nb_muscle (int) : number of muscle in the biorbd model
        - nb_q (int):  number of q in the biorbd model.
        - batch_size (int, optional): Number of entries to buffer before writing to the file. Default is 1000.
        - noise_std_dev (float, optional): Standard deviation of the noise to be added to the data. Default is 0.01.
        """
        self.filename = filename
        self.q_ranges_names_with_dofs = q_ranges_names_with_dofs
        self.nb_muscles = nb_muscles
        self.nb_q = nb_q
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
                **{f"dlmt_dq_{j}_{self.q_ranges_names_with_dofs[k]}": [] for j in range(self.nb_muscles) for k in range(len(self.q_ranges_names_with_dofs)) },
                **{f"muscle_force_{k}": [] for k in range(self.nb_muscles)},
                **{f"torque_{k}": [] for k in range(self.nb_q)},
                 }
            # Create a DataFrame with the initial structure and write it to the CSV file
            pd.DataFrame(data).to_csv(filename, index=False)
    
    def add_line(self, new_line):
        
        """ Add a new line of data to the buffer. If the buffer reaches the batch size, write the buffered data
        to the CSV file.

        Args:
        new_line (dict): A dictionary containing the data to be added to the buffer.
        """
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
    
    def augment_data_with_noise_batch(self, dataset_size_noise):
        """ Augment the existing dataset with additional noisy data to reach the desired dataset size.
        
        Args:
        dataset_size_noise (int): The target size of the dataset including the added noisy data.
        
        Notes:
        - The length of the DataFrame (df) must be a multiple of self.batch_size to ensure correct row additions.
        """
        # Load the dataset from the file, assuming the filename does not include "_with_noise"
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
        """ Add noisy data to a batch of the dataset.

        Args:
        - df_batch (pd.DataFrame): A DataFrame containing a batch of data.
        - num_rows_to_add (int): The number of noisy rows to add to the dataset.
        """
        augmented_data = []
        
        for _ in range(num_rows_to_add):
            # Select a random row from the DataFrame batch, excluding the first column
            row = df_batch.iloc[np.random.randint(len(df_batch)), 1:].values
            
            # Generate and add noise to the selected row
            noise = np.random.normal(0, self.noise_std_dev, row.shape)
            noisy_row = row + noise
            
            # Create a list containing the first column value and the noisy row data
            noisy_row_list = [df_batch.iloc[0, 0]] + noisy_row.tolist()
            augmented_data.append(noisy_row_list)
            
        # Add each noisy row to the data buffer
        for l_idx in augmented_data : 
            self.add_line(l_idx)
