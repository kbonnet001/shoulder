import os
import pandas as pd

class CSVBatchWriter:
    def __init__(self, filename, q_ranges_names_with_dofs, batch_size=100):
        self.filename = filename
        self.q_ranges_names_with_dofs = q_ranges_names_with_dofs
        self.batch_size = batch_size
        self.buffer = []
        
        # Check if file exists, if not create it with initial structure
        if not os.path.exists(filename):
            data = {
                "muscle_selected": [],
                **{f"q_{self.q_ranges_names_with_dofs[k]}": [] for k in range(len(self.q_ranges_names_with_dofs))},
                **{f"qdot_{self.q_ranges_names_with_dofs[k]}": [] for k in range(len(self.q_ranges_names_with_dofs))},
                "alpha": [],
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

    def add_line(self, muscle_selected_index, q, qdot, alpha, origin_muscle, insertion_muscle, segment_length, dlmt_dq, muscle_force, torque):
        # Create a new line with the provided data
        new_line = {
            "muscle_selected": muscle_selected_index,
            **{f"q_{self.q_ranges_names_with_dofs[k]}":  q[k] for k in range(len(q))},
            **{f"qdot_{self.q_ranges_names_with_dofs[k]}":  qdot[k] for k in range(len(qdot))},
            "alpha": alpha,
            "origin_muscle_x": origin_muscle[0],
            "origin_muscle_y": origin_muscle[1],
            "origin_muscle_z": origin_muscle[2],
            "insertion_muscle_x": insertion_muscle[0],
            "insertion_muscle_y": insertion_muscle[1],
            "insertion_muscle_z": insertion_muscle[2],
            "segment_length": segment_length, 
            **{f"dlmt_dq_{self.q_ranges_names_with_dofs[k]}": dlmt_dq[k] for k in range(len(dlmt_dq))},
            "muscle_force": muscle_force,
            "torque": torque
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
