#%%
import torch
import matplotlib.pyplot as plt
import numpy as np

# Lista de nombres de archivos de pesos
model_paths = [
    "best_model_weights_fold_1.pth",
    "best_model_weights_fold_2.pth",
    "best_model_weights_fold_3.pth",
    "best_model_weights_fold_4.pth",
    "best_model_weights_fold_5.pth"
]

# Función para cargar pesos y sesgos de un modelo
def load_weights_and_biases(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    weights = {}
    biases = {}

    for name, param in state_dict.items():
        if "weight" in name:
            weights[name] = param.numpy()
        elif "bias" in name:
            biases[name] = param.numpy()

    return weights, biases

# Cargar pesos y sesgos de todos los modelos
models_weights = []
models_biases = []

for path in model_paths:
    weights, biases = load_weights_and_biases(path)
    models_weights.append(weights)
    models_biases.append(biases)


# Visualización de pesos y sesgos con el modelo promediado
def plot_comparison(models_params, param_type="weights"):
    layers = set()
    # Obtener nombres de capas de todos los modelos
    for params in models_params:
        layers.update(params.keys())
    layers = sorted(layers)

    # Crear subplots para cada capa
    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 5 * len(layers)))
    if len(layers) == 1:
        axes = [axes]  # Asegurar que axes sea una lista si solo hay un subplot

    for ax, layer in zip(axes, layers):
        values = [params.get(layer) for params in models_params]
        # Filtrar valores nulos (si alguna capa no está en todos los modelos)
        values = [v for v in values if v is not None]
        ax.boxplot([v.flatten() for v in values], labels=[f'Model {i+1}' for i in range(len(values))])

        ax.set_title(f"{param_type.capitalize()} comparison for layer: {layer}")
        ax.set_ylabel(f"{param_type.capitalize()} values")

    plt.tight_layout()
    plt.show()

# Visualizar pesos con el modelo promediado
plot_comparison(models_weights, param_type="weights")

# Visualizar sesgos con el modelo promediado
plot_comparison(models_biases, param_type="biases")

# %%
