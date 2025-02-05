import numpy as np
import matplotlib.pyplot as plt
import pathlib
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
theta = np.linspace(0, 2 * np.pi, 300)
perturbations = [
    (
        "x",
        perturbations[:300, 0],
        perturbations[300:, 0],
    ),
    (
        "y",
        perturbations[:300, 1],
        perturbations[300:, 1],
    ),
    (
        "z",
        perturbations[:300, 2],
        perturbations[300:, 2],
    ),
]

output_folder = pathlib.Path(__file__).parent

for axis, perturbation, perturbation_derivative in perturbations:
    plt.plot(theta, perturbation, label=f"{axis} perturbation")
    plt.plot(theta, perturbation_derivative, label=f"{axis} perturbation derivative")
    plt.plot(
        theta,
        np.gradient(perturbation, theta),
        label=f"{axis} perturbation numerical derivative",
    )
    plt.title("$\sigma=0.05, l=0.2$")
    plt.xlabel("$\\theta$")
    plt.legend()
    plt.savefig(output_folder / f"{axis}_perturbation.png")
    plt.close()
