import pandas as pd
import numpy as np

class ExcelBatchWriterWithNoise:
    def __init__(self, filename, batch_size=1000, noise_std_dev=0.01):
        self.filename = filename
        self.batch_size = batch_size
        self.noise_std_dev = noise_std_dev
        self.buffer = []
        
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
        
        self.buffer.extend(augmented_data)
        
        if len(self.buffer) >= self.batch_size:
            self._flush()
    
    def _flush(self):
        if not self.buffer:
            return
        
        # Lire les données existantes du fichier Excel
        try:
            df_existing = pd.read_excel(self.filename)
        except FileNotFoundError:
            df_existing = pd.DataFrame()  # Crée un DataFrame vide si le fichier n'existe pas
        
        # Convertir le buffer en DataFrame
        buffer_df = pd.DataFrame(self.buffer, columns=df_existing.columns)
        
        # Vérifier si le DataFrame n'est pas vide ou entièrement NaN
        if not buffer_df.empty and not buffer_df.isna().all().all():
            # Concaténer le DataFrame existant avec le buffer
            df_updated = pd.concat([df_existing, buffer_df], ignore_index=True)
            
            # Écrire le DataFrame mis à jour dans le fichier Excel
            with pd.ExcelWriter(self.filename, engine='openpyxl', mode='w') as writer:
                df_updated.to_excel(writer, index=False)
        
        # Vider le buffer
        self.buffer = []
    
    def augment_data_with_noise_batch(self, num_rows_to_add):
        df = pd.read_excel(self.filename)
        num_chunks = len(df) // self.batch_size + 1
        
        for i in range(num_chunks):
            chunk = df[i*self.batch_size : (i+1)*self.batch_size]
            self.add_noise_to_batch(chunk, num_rows_to_add)
        
        # Flush remaining data in buffer
        self._flush()

