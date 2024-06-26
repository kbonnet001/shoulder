import os
import pandas as pd

class ExcelBatchWriter:
    def __init__(self, filename, batch_size=100):
        self.filename = filename
        self.batch_size = batch_size
        self.buffer = []
        
        # Check if file exists, if not create it with initial structure
        if not os.path.exists(filename):
            data = {
                "muscle_selected": [],
                "humerus_right_RotY": [],
                "humerus_right_RotX": [],
                "humerus_right_RotY2": [],
                "ulna_effector_right_RotZ": [],
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
            "humerus_right_RotY": q[0],
            "humerus_right_RotX": q[1],
            "humerus_right_RotY2": q[2],
            "ulna_effector_right_RotZ": q[3],
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
        
        # Read the existing data from the Excel file
        df = pd.read_excel(self.filename)
        
        # Append the buffered lines to the DataFrame
        df = pd.concat([df, pd.DataFrame(self.buffer)], ignore_index=True)
        
        # Write the updated DataFrame back to the Excel file
        with pd.ExcelWriter(self.filename, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, index=False)
        
        # Clear the buffer
        self.buffer = []

    def close(self):
        # Flush any remaining lines in the buffer when closing
        self._flush()
