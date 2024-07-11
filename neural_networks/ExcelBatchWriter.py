import os
import pandas as pd

class ExcelBatchWriter:
    def __init__(self, filename, q_ranges_names_with_dofs, batch_size=100):
        self.filename = filename
        self.q_ranges_names_with_dofs = q_ranges_names_with_dofs
        self.batch_size = batch_size
        self.buffer = []
        
        # Check if file exists, if not create it with initial structure
        if not os.path.exists(filename):
            data = {
                "muscle_selected": [],
                **{self.q_ranges_names_with_dofs[k]: [] for k in range(len(self.q_ranges_names_with_dofs))},
                "origin_muscle_x": [],
                "origin_muscle_y": [],
                "origin_muscle_z": [],
                "insertion_muscle_x": [],
                "insertion_muscle_y": [],
                "insertion_muscle_z": [],
                "segment_length": []
                 }
            pd.DataFrame(data).to_excel(filename, index=False)

    def add_line(self, muscle_selected_index, q, origin_muscle, insertion_muscle, segment_length):
        # Create a new line with the provided data
        new_line = {
            "muscle_selected": muscle_selected_index,
            **{self.q_ranges_names_with_dofs[k]: q[k] for k in range(len(q))},
            "origin_muscle_x": origin_muscle[0],
            "origin_muscle_y": origin_muscle[1],
            "origin_muscle_z": origin_muscle[2],
            "insertion_muscle_x": insertion_muscle[0],
            "insertion_muscle_y": insertion_muscle[1],
            "insertion_muscle_z": insertion_muscle[2],
            "segment_length": segment_length
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
