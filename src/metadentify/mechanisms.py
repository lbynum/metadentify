from collections.abc import Callable
from copy import deepcopy
from typing import AnyStr

import dowhy
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from dowhy import gcm
from scipy import stats


class BaseRandomPredictionModel(gcm.PredictionModel):
    def fit(self, X, Y):
        pass

    def clone(self):
        return deepcopy(self)


class GroundTruthPredictionModel(gcm.PredictionModel):
    def __init__(self, prediction_function: Callable[[np.ndarray], float | np.ndarray]) -> None:
        super().__init__()
        self.prediction_function = prediction_function

    def fit(self, X, Y):
        pass

    def predict(self, X):
        Y = self.prediction_function(X)
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        return Y

    def clone(self):
        return GroundTruthPredictionModel(prediction_function=deepcopy(self.prediction_function))


def build_synthetic_causal_model(
    causal_graph: nx.DiGraph,
    node_function_dict: dict[
        AnyStr,
        tuple[BaseRandomPredictionModel | Callable | None, gcm.ScipyDistribution],
    ],
) -> gcm.InvertibleStructuralCausalModel:
    causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)
    for node in causal_graph.nodes:
        noise_model = node_function_dict[node][1]

        if noise_model is None:
            noise_model = gcm.ScipyDistribution(stats.norm, loc=0, scale=0)

        if not noise_model._fixed_parameters:
            raise ValueError(f"Noise model for node {node} is not parameterized.")

        node_parents = list(causal_graph.predecessors(node))
        if len(node_parents) > 0:
            prediction_function = node_function_dict[node][0]
            if isinstance(prediction_function, gcm.PredictionModel):
                prediction_model = prediction_function
            else:
                prediction_model = GroundTruthPredictionModel(
                    prediction_function=prediction_function
                )
            current_mechanism = gcm.AdditiveNoiseModel(
                prediction_model=prediction_model, noise_model=noise_model
            )
        else:
            current_mechanism = noise_model

        causal_model.set_causal_mechanism(node, current_mechanism)

        causal_model.graph.nodes[node]["parents_during_fit"] = dowhy.graph.get_ordered_predecessors(
            causal_graph=causal_model.graph, node=node
        )

    return causal_model


class NeuralNetMechanism(BaseRandomPredictionModel):
    def __init__(
        self,
        weights_mean: float = 0.0,
        weights_std: float = 1.0,
        hidden_dim: int = 10,
        num_layers: int = 2,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.weights_mean = weights_mean
        self.weights_std = weights_std
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.activation_1 = activation if activation is not None else nn.PReLU()

        self._model = None

    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=self.weights_mean, std=self.weights_std)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            X = X.reshape((-1, 1))

        n_inputs = X.shape[1]
        n_samples = X.shape[0]

        if self._model is None:
            layers = []
            in_dim = n_inputs
            for _ in range(self.num_layers):
                layers.append(nn.Linear(in_dim, self.hidden_dim))
                layers.append(deepcopy(self.activation_1))
                layers.append(nn.LayerNorm(self.hidden_dim))
                in_dim = self.hidden_dim
            layers.append(nn.Linear(in_dim, 1))
            self._model = nn.Sequential(*layers)
            self._model.apply(self._weight_init)

        data = X.astype("float32")
        data = torch.from_numpy(data)

        effect = np.reshape(self._model(data).data.numpy(), (n_samples,))

        return effect


class RandomLinearMechanism(BaseRandomPredictionModel):
    def __init__(
        self,
        weights_min: float = -1.0,
        weights_max: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.weights_min = weights_min
        self.weights_max = weights_max
        self.bias = bias

        self._model = None

    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight.data, a=self.weights_min, b=self.weights_max)
            if module.bias is not None:
                nn.init.uniform_(module.bias.data, a=self.weights_min, b=self.weights_max)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            X = X.reshape((-1, 1))

        n_inputs = X.shape[1]
        n_samples = X.shape[0]

        if self._model is None:
            self._model = nn.Linear(n_inputs, 1, bias=self.bias)
            self._model.apply(self._weight_init)

        data = X.astype("float32")
        data = torch.from_numpy(data)

        effect = np.reshape(self._model(data).data.numpy(), (n_samples,))

        return effect
