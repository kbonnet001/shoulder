import os
import matplotlib.pyplot as plt

def create_directory(directory_path):
    """
    create a new folder

    INPUT :
        directory_path (str): path of the new folder
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"The folder'{directory_path}' have been created.")
    else:
        print(f"The folder '{directory_path}' already exist.")

def create_and_save_plot(directory_path, file_name):
    """
    Save a Matplotlib fig in a specific folder

    INPUTS :
        directory_path (str): Path where the fig will be save.
        file_name (str): Name of the figure
    """
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    # Enregistre la figure
    file_path = os.path.join(directory_path, file_name)
    plt.savefig(file_path)

def save_text_to_file(text, file_path):
    """_summary_

    Args:
        text (_type_): _description_
        file_path (_type_): _description_
    """
    with open(file_path, 'w') as file:
        file.write(text)
