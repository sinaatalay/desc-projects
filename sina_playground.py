import numpy as np
import jax
import jax.numpy as jnp
import sympy as sp
import dataclasses
import functools


@dataclasses.dataclass
class StochasticOptimizationSettings:
    """See https://doi.org/10.1088/1741-4326/ac45f3 for implementation details
    of the stochastic coil optimization.

    For the covariance function, a squared exponential function is used:

    k(d) = sigma^2 * exp(-(d)^2 / (2 * l^2))
    cov_function(d) = SUM(i=-inf to inf) k(d + 2*pi*i)

    where d is (theta_1-theta_2), sigma is the standard deviation and l is the
    length_scale. Note that k isn't used directly as the covarince function
    (kernel), but an infinite sum is used to make it peroiodic on [0, 2*pi).

    It's later used to construct the covariance matrix K. It will allow us to
    draw random samples from a multivariate normal distribution with mean 0 and
    covariance K. It's a (n*2) x (n*2) matrix, where n is the number of points
    to perturb. For each point, we will draw 2 random numbers, one for the
    position perturbation and one for the tangent perturbation. For each point,
    we will do this drawing 3 times, so we have the perturnations for 3 spatial
    dimensions.

    Parameters
    ----------
    number_of_samples : int
        Number of "perturbed" coils to include in the objective.
    standard_deviation : float
        The standart deviation (sigma) and the characteristic length (l) used in
        the covariance function.
    length_scale : float
        The standart deviation (sigma) and the characteristic length (l) used in
        the covariance function.
    seed : int, optional
        Seed for the pseudo-random number generator. Defaults to 0.
    """

    number_of_samples: int
    standard_deviation: float
    length_scale: float
    seed: int = 0
    # The fields below are not supposed to be set by the user, they will be derived
    # in the `build` method.
    number_of_discretization_points: int = 0
    zero_mean_array: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.zeros(10)
    )
    index_array_for_samples: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.zeros(10)
    )
    covariance_matrix: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.zeros((10, 10))
    )

    def compute_covariance_matrix(self) -> jnp.ndarray:
        # Sympy is used here because we need a derivative. This is a one time
        # computation and then it's cached with `functools.cached_property`. So
        # I don't think it's worth to optimize this part.
        theta_1, theta_2, d, i = sp.symbols("theta_1 theta_2 d i")
        covariance_function = sp.Sum(
            (
                self.standard_deviation**2
                * sp.exp(-((d + i * 2 * sp.pi) ** 2) / (2 * self.length_scale**2))
            ),
            (i, -6, 6),
        )

        # Covariance between two different position perturbations:
        cov_f_pp = sp.lambdify(
            (theta_1, theta_2),
            covariance_function.subs(d, theta_1 - theta_2),
            "numpy",
        )
        # Covariance between a position perturbation derivartive and a position
        # perturbation:
        cov_f_dp = sp.lambdify(
            (theta_1, theta_2),
            -sp.diff(covariance_function, d).subs(d, theta_1 - theta_2),
            "numpy",
        )
        # Covariance between a position perturbation and a position perturbation
        # derivartive:
        cov_f_pd = sp.lambdify(
            (theta_1, theta_2),
            -sp.diff(covariance_function, d).subs(d, theta_1 - theta_2),
            "numpy",
        )
        # Covariance between a position perturbation derivartive and a position
        # perturbation derivartive:
        cov_f_dd = sp.lambdify(
            (theta_1, theta_2),
            -sp.diff(covariance_function, d, 2).subs(d, theta_1 - theta_2),
            "numpy",
        )

        # Construct 2n x 2n covariance matrix:
        # K = [[cov_f_pp, cov_f_pd], [cov_f_dp, cov_f_dd]]
        thetas = jnp.linspace(
            0, 2 * jnp.pi, self.number_of_discretization_points, endpoint=False
        )
        XX, YY = jnp.meshgrid(thetas, thetas)
        return jnp.block(
            [
                [cov_f_pp(XX, YY), cov_f_pd(XX, YY)],
                [cov_f_dp(XX, YY), cov_f_dd(XX, YY)],
            ]
        )

    @functools.cached_property
    def perturbations(self) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        # Draw a random matrix shaped (2n, 3) from a multivariate normal
        # distribution, where the top half is for the position perturbations and
        # the bottom half is for the tangent perturbations, using the covariance
        # matrix and zero mean.

        mean_1d = self.zero_mean_array

        keyx, keyy, keyz = jax.random.split(jax.random.PRNGKey(self.seed), 3)

        # Each draw is shape (2n,)
        xdraw = jax.random.multivariate_normal(keyx, mean_1d, self.covariance_matrix)
        ydraw = jax.random.multivariate_normal(keyy, mean_1d, self.covariance_matrix)
        zdraw = jax.random.multivariate_normal(keyz, mean_1d, self.covariance_matrix)

        # Create a (2n,3) array for the position and tangent perturbations. The first
        # n rows are for the position perturbations and the second n rows are for the
        # tangent perturbations.
        perturbations = jnp.stack(
            [
                xdraw,
                ydraw,
                zdraw,
            ],
            axis=1,
        )

        return perturbations


perturbations = StochasticOptimizationSettings(
    number_of_samples=1,
    length_scale=1,
    standard_deviation=0.1,
    number_of_discretization_points=100,
    zero_mean_array=np.zeros(2 * 100),
)
perturbations.covariance_matrix = perturbations.compute_covariance_matrix()

this_is_an_an_ndarray_with_all_nans = perturbations.perturbations
