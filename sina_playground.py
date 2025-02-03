import numpy as np
import matplotlib.pyplot as plt
from desc.objectives._coils import StochasticOptimizationSettings

perturbations = StochasticOptimizationSettings(
    number_of_samples=1,
    length_scale=0.2,
    standard_deviation=0.05,
    number_of_discretization_points=300,
    zero_mean_array=np.zeros(2 * 300),
)
perturbations.covariance_matrix = perturbations.compute_covariance_matrix()
perturbations = perturbations.perturbations
# compute magnitude of each row:
# perturbations = np.linalg.norm(perturbations, axis=1)
# plot half of the parturbations from 0 to 2pi
x = np.linspace(0, 2 * np.pi, 300)
x_perturbation = perturbations[:300, 2]
x_perturbation_derivative_numerical = np.gradient(x_perturbation, x)
x_perturbation_derivative = perturbations[300:, 2]

plt.plot(x, x_perturbation, label="z perturbation")
plt.plot(x, x_perturbation_derivative, label="z perturbation derivative")
plt.plot(
    x, x_perturbation_derivative_numerical, label="z perturbation numerical derivative"
)
plt.legend()
# title:
plt.title("$\sigma=0.05, l=0.2$")
# axis labels:
plt.xlabel("$\\theta$")
plt.show()
