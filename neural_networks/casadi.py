######
# Gros chantier, mis de cote pour le moment car pas prioritaire pour le stage
######

# import casadi as ca
# import numpy as np
# import torch
# import torch.nn as nn
# from neural_networks.save_model import load_saved_model
# from neural_networks.Model import Model

# # a mettre dans save apres


# def pytorch_to_casadi(model, input_shape):
#     """

#     INPUTS :
#         model : model loaded in eval mode
#         input_shape (_type_): _description_

#     OUTPUTS :
#         _type_: _description_
#     """
#     # Définir les variables CasADi
#     x = ca.MX.sym('x', input_shape)
    
#     # Convertir la fonction PyTorch en fonction CasADi
#     def forward_fn(x_np):
#         x_tensor = torch.tensor(x_np, dtype=torch.float32)
#         with torch.no_grad():
#             y_tensor = model(x_tensor)
#         return y_tensor.numpy()
    
#     y = ca.MX(np.vectorize(forward_fn, signature='(n)->(m)')(x))
#     return ca.Function('model', [x], [y])

# # def pytorch_to_casadi(model, input_shape):
# #     """
# #     Converts a PyTorch model to a CasADi function.
    
# #     INPUTS :
# #         model : PyTorch model loaded in eval mode
# #         input_shape (tuple): Shape of the input tensor
    
# #     OUTPUTS :
# #         casadi.Function: CasADi function equivalent of the PyTorch model
# #     """
# #     # Define CasADi variables
# #     x = ca.MX.sym('x', input_shape)
    
# #     # Convert PyTorch function to CasADi function
# #     def forward_fn(x_np):
# #         x_tensor = torch.tensor(x_np, dtype=torch.float32)
# #         with torch.no_grad():
# #             y_tensor = model(x_tensor)
# #         return y_tensor.numpy()
    
# #     # Create a wrapper function that uses CasADi's `Function` to call `forward_fn`
# #     def casadi_forward_fn(x_casadi):
# #         x_np = np.array(x_casadi)
# #         y_np = forward_fn(x_np)
# #         return y_np
    
# #     y = ca.MX(casadi_forward_fn(x))
# #     casadi_model = ca.Function('model', [x], [y])
    
# #     return casadi_model


# # def pytorch_to_casadi(model, input_shape):
# #     """
# #     INPUTS :
# #         model : model loaded in eval mode
# #         input_shape (_type_): _description_

# #     OUTPUTS :
# #         _type_: _description_
# #     """
# #     # Définir les variables CasADi
# #     x = ca.MX.sym('x', input_shape)
    
# #     # Fonction pour convertir l'entrée CasADi en entrée PyTorch et obtenir la sortie
# #     def forward_fn(x_casadi):
# #         x_np = ca.DM(x_casadi).full()  # Convertir CasADi DM en numpy array
# #         x_tensor = torch.tensor(x_np, dtype=torch.float32)
# #         with torch.no_grad():
# #             y_tensor = model(x_tensor)
# #         y_np = y_tensor.numpy()
# #         return y_np

# #     # Utiliser un map pour appliquer la fonction sur chaque élément de x
# #     y = ca.DM.forward_fn(x)
# #     return ca.Function('model', [x], [y])


# # def pytorch_to_casadi(model, input_shape):
# #     """
# #     INPUTS :
# #         model : model loaded in eval mode
# #         input_shape (_type_): _description_

# #     OUTPUTS :
# #         _type_: _description_
# #     """
# #     # Définir les variables CasADi
# #     x = ca.MX.sym('x', input_shape)
    
# #     # Convertir la fonction PyTorch en fonction CasADi
# #     def forward_fn(x_casadi):
# #         x_np = np.array(x_casadi).reshape(input_shape)  # Convertir CasADi MX en numpy array
# #         x_tensor = torch.tensor(x_np, dtype=torch.float32)
# #         with torch.no_grad():
# #             y_tensor = model(x_tensor)
# #         y_np = y_tensor.numpy()
# #         return y_np

# #     # Créer une fonction CasADi pour appliquer forward_fn
# #     y = ca.MX.zeros((input_shape[0],))  # Définir la taille de y en fonction de la sortie attendue
# #     for i in range(input_shape[0]):
# #         y[i] = ca.Function('forward_fn', [x], [ca.MX(forward_fn(x))])(x)[i]
    
# #     return ca.Function('model', [x], [y])

# # # Convertir le modèle PyTorch en fonction CasADi
# # input_shape = (input_size,)  # ou la forme correcte de vos entrées
# # casadi_model = pytorch_to_casadi(model, input_shape)

# # casadi_model = ca.Function('model', [x], [y])
# # # fonction finale que l'on veut
# # # Exemples d'utilisation de la fonction CasADi
# # input_data = np.random.rand(input_size)
# # output_data = casadi_model(input_data)

# # print(output_data)

# def load_model_to_casadi(file_path_model, input_shape) : 
#     """

    
#     Args:
#         file_path_model : string, path where the file 'model_config.json' of the model could be find
#         input_data (_type_): _description_
#     """
#     model = load_saved_model(file_path_model)
    
#     return pytorch_to_casadi(model, input_shape)


# ### 

# # file_path_model = 'data_generation_datas_with_dlmt_dq/PECM2/essai_muscle_best'
# # input_shape = 8

# # casadi_model_test = load_model_to_casadi(file_path_model, input_shape)

# # input_data = 
# # output_data = casadi_model_test(input_data)

# import torch

# def get_weights_biases(model):
#     weights = []
#     biases = []
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             weights.append(param.detach().numpy())
#         elif 'bias' in name:
#             biases.append(param.detach().numpy())
#     return weights, biases

# import casadi as ca

# # def gelu(x):
# #     return x * (0.5 * (1 + ca.tanh(ca.sqrt(2 / ca.M_PI) * (x + 0.044715 * ca.pow(x, 3)))))

# # def gelu(x):
# #     pi = np.pi
# #     return x * (0.5 * (1 + ca.tanh(ca.sqrt(2 / pi) * (x + 0.044715 * ca.pow(x, 3)))))

# def gelu(x):
#     pi = 3.141592653589793
#     x2 = ca.mtimes(x, x)  # x^2
#     x3 = ca.mtimes(x2, x)  # x^3
#     sqrt2_over_pi = ca.sqrt(2 / pi)
#     term = sqrt2_over_pi * (x + 0.044715 * x3)
#     return x * (0.5 * (1 + ca.tanh(term)))

# def pytorch_to_casadi(weights, biases, activations, input_size, output_size):
#     x = ca.MX.sym('x', input_size)
#     for i in range(len(weights)):
#         w = weights[i]
#         b = biases[i]
#         x = ca.mtimes(w, x) + b
#         if activations[i] == 'ReLU':
#             x = ca.fmax(x, 0)
#         elif activations[i] == 'Sigmoid':
#             x = 1 / (1 + ca.exp(-x))
#         elif activations[i] == 'GELU':
#             x = gelu(x)
#         else : 
#             print("None activation correspond")
#     return x

# #---------------------
# import torch
# import torch.nn as nn
# import casadi as ca
# import numpy as np

# def casadi_test(model, x_value, input_size) : 

#     # Définir une fonction pour la prédiction en utilisant le modèle PyTorch
#     def pytorch_predict(x):
#         with torch.no_grad():
#             x_tensor = torch.tensor(x, dtype=torch.float32)
#             y_tensor = model(x_tensor)
#             return y_tensor.numpy()

#     # Définir une fonction CasADi à partir de la fonction de prédiction
#     def casadi_predict(x):
#         # x = ca.MX(x)  # Assurez-vous que x est de type CasADi MX
#         # x = ca.reshape(x, (1, -1))  # Reshape using CasADi function
#         # x_num = np.array(ca.DM(x)).flatten()  # Convertir en NumPy array en utilisant CasADi DM
#         # x_num = np.array(x).flatten()  # Convertir MX en array NumPy
#         x_num = x
#         y = pytorch_predict(x_num)
#         return ca.vertcat(*y)

#     # Créer une fonction CasADi
#     x = ca.MX.sym('x', input_size)
#     y = casadi_predict(x)
#     f = ca.Function('f', [x], [y])

#     # Exemple d'utilisation de la fonction CasADi
#     x_value = np.random.randn(input_size)
#     y_value = f(x_value)
#     # print(f'Input: {x_value}\nOutput: {y_value}')