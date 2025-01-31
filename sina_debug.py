# # Stage-Two Filamentary Coil Optimization
# This notebook will show how to use DESC to perform stage two coil optimization with filamentary coils of differing parameterizations, such as Fourier in terms of arbitrary angle (as pioneered by [FOCUS][1]) or planar coils described in terms of the coil center, normal to the plane, and a Fourier series describing the radius of the coil in that plane. We will first find a coilset for the precise QA ([Landreman & Paul 2022][2]) vacuum equilibrium, starting from an initial coilset composed of circular planar coils, with only coils parameterized by a Fourier series in an arbitrary curve parameter. A second coil optimization will be performed for a W7-X-like finite beta equilibrium using a coilset composed of various coil parameterizations. Once optimized, the normal field error will be assessed, and for the vacuum equilibrium, field line tracing will be performed to compare the Poincare trace to the equilibrium flux surfaces.
#
#
# [1]: <https://doi.org/10.1088/1741-4326/aa8e0a> "Zhu, C., Hudson, S. R., Song, Y. & Wan, Y. New method to design stellarator coils without the winding surface. Nucl. Fusion 58, 016008 (2018)."
#
# [2]: <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.035001> "M. Landreman. & E. Paul. 2022 Physical Review Letters"


import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))


# from desc import set_device
# set_device("gpu")


import numpy as np
from desc.coils import CoilSet, FourierPlanarCoil
import desc.examples
from desc.equilibrium import Equilibrium
from desc.plotting import plot_2d, plot_3d, plot_coils, plot_comparison
from desc.grid import LinearGrid
from desc.coils import MixedCoilSet
from desc.objectives import (
    ObjectiveFunction,
    CoilCurvature,
    CoilLength,
    CoilTorsion,
    CoilSetMinDistance,
    PlasmaCoilSetMinDistance,
    QuadraticFlux,
    ToroidalFlux,
    FixCoilCurrent,
    FixParameters,
)
from desc.objectives._coils import StochasticOptimizationSettings
from desc.optimize import Optimizer
from desc.integrals import compute_B_plasma
import time


# ## Coil Optimization Metrics


# Two main figures of merit regarding the quality of the coil optimization in recreating the equilibrium are the average normalized magnetic field and Poincare plot from the coils. The average normalized magnetic field is a measure of how well the coils' field (plus the plasma field if it is finite beta) recreates last closed flux surface of the target equilibrium and is given as
# $$
# \frac{\left<|\mathbf{B}_{\text{total}}\cdot \mathbf{\hat{n}}|\right>}{\left<|\mathbf{B}_{\text{total}}|\right>}
# $$
# where $\mathbf{\hat{n}}$ is the normal vector to the equilibrium's last closed flux surface and $\mathbf{B}_{\text{total}} = \mathbf{B}_{\text{coils}} + \mathbf{B}_{\text{plasma currents}}$. Field traces are also significant for vacuum equilibria because they visually show the quality of the coil field in recreating the equilibrium vacuum flux surfaces.
#
# These two metrics are calculated in functions below.


def compute_average_normalized_field(field, eq, vacuum=False):
    grid = LinearGrid(M=80, N=80, NFP=eq.NFP)
    Bn, surf_coords = field.compute_Bnormal(eq, eval_grid=grid)
    normalizing_field_vec = field.compute_magnetic_field(surf_coords)
    if not vacuum:
        # add plasma field to the normalizing field
        normalizing_field_vec += compute_B_plasma(eq, eval_grid=grid)
    normalizing_field = np.mean(np.linalg.norm(normalizing_field_vec, axis=1))
    return np.mean(np.abs(Bn)) / normalizing_field


def plot_field_lines(field, eq):
    # for starting locations we'll pick positions on flux surfaces on the outboard midplane
    grid_trace = LinearGrid(rho=np.linspace(0, 1, 9))
    r0 = eq.compute("R", grid=grid_trace)["R"]
    z0 = eq.compute("Z", grid=grid_trace)["Z"]
    fig, ax = desc.plotting.plot_surfaces(eq)
    fig, ax = desc.plotting.poincare_plot(
        field,
        r0,
        z0,
        NFP=eq.NFP,
        ax=ax,
        color="k",
        size=1,
    )
    return fig, ax


# ## Vacuum
#
# We will be focusing on optimizing coils with the vacuum precise QA as the target equilibrium.


eq = desc.examples.get("precise_QA")


# ### Make initial coilset
#
# We start by creating planar coils centered and aligned with the magnetic axis, equally spaced in the toroidal angle phi. Note that the coil positions are chosen to avoid the symmetry planes. We then convert the planar coils to the FourierXYZ parameterization. There are only 3 "unique" coils, and the full CoilSet accounts for the other "virtual" coils from field period and stellarator symmetry.


minor_radius = eq.compute("a")["a"]
offset = 0.3
num_coils = 3  # coils per half field period

zeta = np.linspace(0, np.pi / eq.NFP, num_coils, endpoint=False) + np.pi / (
    2 * eq.NFP * num_coils
)
grid = LinearGrid(rho=[0.0], M=0, zeta=zeta, NFP=eq.NFP)
data = eq.axis.compute(["x", "x_s"], grid=grid, basis="rpz")

centers = data["x"]  # center coils on axis position
normals = data["x_s"]  # make normal to coil align with tangent along axis

unique_coils = []
for k in range(num_coils):
    coil = FourierPlanarCoil(
        current=1e6,
        center=centers[k, :],
        normal=normals[k, :],
        r_n=minor_radius + offset,
        basis="rpz",  # we are giving the center and normal in cylindrical coordinates
    ).to_FourierXYZ(
        N=10
    )  # fit with 10 fourier coefficients per coil
    unique_coils.append(coil)

# We package these coils together into a CoilSet, which has efficient methods for calculating
# the total field while accounting for field period and stellarator symmetry
# Note that `CoilSet` requires all the member coils to have the same parameterization and resolution.
# if we wanted to use coils of different types or resolutions, we can use a `MixedCoilSet` (see the next section below)
coilset = CoilSet(unique_coils, NFP=eq.NFP, sym=eq.sym)

coil: FourierPlanarCoil
# perturbations = StochasticOptimizationSettings(
#     length_scale=1, standard_deviation=0.1, number_of_discretization_points=coil.zeta
# )

# visualize the initial coilset
# we use a smaller than usual plot grid to reduce memory of the notebook file
plot_grid = LinearGrid(M=20, N=40, NFP=1, endpoint=True)
fig = plot_3d(eq, "|B|", grid=plot_grid)
fig = plot_coils(coilset, fig=fig)
fig.show()


# ### Optimizing a `FourierXYZCoil` coil set


# number of points used to discretize coils. This could be different for each objective
# (eg if you want higher resolution for some calculations), but we'll use the same for all of them
coil_grid = LinearGrid(N=50)
# similarly define a grid on the plasma surface where B*n errors will be evaluated
plasma_grid = LinearGrid(M=25, N=25, NFP=eq.NFP, sym=eq.sym)


# define our objective function (we will use a helper function here to make it easier to change weights later)
weights = {
    "quadratic flux": 200,
    "coil-coil min dist": 100,
    "plasma-coil min dist": 10,
    "coil curvature": 500,
    "coil length": 2,
}


def make_vac_coil_obj(eq, coilset, weights_dict):
    obj = ObjectiveFunction(
        (
            QuadraticFlux(
                eq,
                field=coilset,
                # grid of points on plasma surface to evaluate normal field error
                eval_grid=plasma_grid,
                field_grid=coil_grid,
                vacuum=True,  # vacuum=True means we won't calculate the plasma contribution to B as it is zero
                weight=weights_dict["quadratic flux"],
                stochastic_optimization_settings={
                    "number_of_samples": 10,
                    "standard_deviation": 0.01,
                    "length_scale": 0.005,
                },  # 5 mm scale length
            ),
            CoilSetMinDistance(
                coilset,
                # in normalized units, want coil-coil distance to be at least 10% of minor radius
                bounds=(0.1, np.inf),
                normalize_target=False,  # we're giving bounds in normalized units
                grid=coil_grid,
                weight=weights_dict["coil-coil min dist"],
            ),
            PlasmaCoilSetMinDistance(
                eq,
                coilset,
                # in normalized units, want plasma-coil distance to be at least 25% of minor radius
                bounds=(0.25, np.inf),
                normalize_target=False,  # we're giving bounds in normalized units
                plasma_grid=plasma_grid,
                coil_grid=coil_grid,
                eq_fixed=True,  # Fix the equilibrium. For single stage optimization, this would be False
                weight=weights_dict["plasma-coil min dist"],
            ),
            CoilCurvature(
                coilset,
                # this uses signed curvature, depending on whether it curves towards
                # or away from the centroid of the curve, with a circle having positive curvature.
                # We give the bounds normalized units, curvature of approx 1 means circular,
                # so we allow them to be a bit more strongly shaped
                bounds=(-1, 2),
                normalize_target=False,  # we're giving bounds in normalized units
                grid=coil_grid,
                weight=weights_dict["coil curvature"],
            ),
            CoilLength(
                coilset,
                bounds=(0, 2 * np.pi * (minor_radius + offset)),
                normalize_target=True,  # target length is in meters, not normalized
                grid=coil_grid,
                weight=weights_dict["coil length"],
            ),
        )
    )
    return obj


obj = make_vac_coil_obj(eq, coilset, weights)
## define our constraints
# we will constrain the current of one coil to avoid the trivial quadratic flux minimizing solution
# of zero coil current
coil_indices_to_fix_current = [False for c in coilset]
coil_indices_to_fix_current[0] = True
constraints = (FixCoilCurrent(coilset, indices=coil_indices_to_fix_current),)

# Alternatively, we could have used
# constraints = (ToroidalFlux(eq, coilset, eq_fixed=True),)
# to target the actual toroidal flux from the equilibrium, or
# constraints = (FixSumCoilCurrent(coilset),)
# to fix the sum of all the coil currents but as this is a vacuum calculation
# the specific value of the flux doesn't matter, we can always rescale the coil currents later.

# Pick an optimizer. For this problem with only linear constraints we can use a regular least squares method.
# if we used the ToroidalFlux constraint we would need to use a constrained optimization method
optimizer = Optimizer("lsq-exact")

(optimized_coilset,), _ = optimizer.optimize(
    coilset,
    objective=obj,
    constraints=constraints,
    maxiter=20,
    verbose=3,
    copy=True,
    ftol=1e-4,
)


normalized_field = compute_average_normalized_field(optimized_coilset, eq, vacuum=True)
print(f"<Bn> = {normalized_field:.3e}")


## visualize the optimized coilset and the normal field error
# passing in eq.surface avoids the unnecessary plasma field computation (as this is a vacuum eq)
fig = plot_3d(
    eq.surface, "B*n", field=optimized_coilset, field_grid=coil_grid, grid=plot_grid
)

fig = plot_coils(optimized_coilset, fig=fig)
fig.show()


plot_field_lines(optimized_coilset, eq)
