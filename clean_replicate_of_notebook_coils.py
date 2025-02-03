
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
from desc.coils import (
    CoilSet,
    FourierPlanarCoil,
    MixedCoilSet,
    initialize_modular_coils,
    initialize_saddle_coils,
)
import desc.examples
from desc.equilibrium import Equilibrium
from desc.plotting import plot_2d, plot_3d, plot_coils, plot_comparison
from desc.grid import LinearGrid
from desc.objectives import (
    ObjectiveFunction,
    CoilCurvature,
    CoilLength,
    CoilTorsion,
    CoilSetLinkingNumber,
    CoilSetMinDistance,
    PlasmaCoilSetMinDistance,
    QuadraticFlux,
    ToroidalFlux,
    FixCoilCurrent,
    FixParameters,
)
from desc.optimize import Optimizer
from desc.integrals import compute_B_plasma
import time
import plotly.express as px
import plotly.io as pio

# This ensures Plotly output works in multiple places:
# plotly_mimetype: VS Code notebook UI
# notebook: "Jupyter: Export to HTML" command in VS Code
# See https://plotly.com/python/renderers/#multiple-renderers
pio.renderers.default = "plotly_mimetype+notebook"


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


coilset = initialize_modular_coils(eq, num_coils=3, r_over_a=3.0).to_FourierXYZ()

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
                bounds=(0, 2 * np.pi * (coilset[0].compute("length")["length"])),
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
    maxiter=100,
    verbose=3,
    ftol=1e-4,
    copy=True,
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


plot_field_lines(optimized_coilset, eq);


# ## Using REGCOIL Algorithm for initial guess


# Instead of starting from a circular coilset, we can do a sort of "warm-start" by first solving a simpler problem where we find a surface current on an exterior surface which minimizes the normal field error. This problem can be quickly solved as a regularized least-squares problem ([See the REGCOIL tutorial notebook for more info](./coil_optimization_REGCOIL.ipynb)), which can give an initial guess coilset which has low Bn from which to further refine with filamentary coil optimization.


from desc.magnetic_fields import (
    solve_regularized_surface_current,
    FourierCurrentPotentialField,
)
from scipy.constants import mu_0

# First, make a constant-offset surface upon which our surface current will lie.
surf = eq.surface.constant_offset_surface(
    offset=0.3,  # desired offset (m)
    M=2,  # Poloidal resolution of desired offset surface
    N=eq.N,  # Toroidal resolution of desired offset surface
    grid=LinearGrid(M=8, N=2 * eq.N, NFP=eq.NFP),
)


plot_comparison([surf, eq], labels=["surf", "eq"], theta=0, rho=np.array(1.0));


# create the FourierCurrentPotentialField object from the constant offset surface we found in the previous cell
surface_current_field = FourierCurrentPotentialField.from_surface(
    surf,
    I=0,
    # manually setting G to value needed to provide the equilibrium's toroidal flux,
    # though this is not necessary as it gets set automatically inside the solve_regularized_surface_current function
    G=np.asarray(
        [
            -eq.compute("G", grid=LinearGrid(rho=np.array(1.0)))["G"][0]
            / mu_0
            * 2
            * np.pi
        ]
    ),
    # set symmetry of the current potential, "sin" is usually expected for stellarator-symmetric surfaces and equilibria
    sym_Phi="sin",
)

surface_current_field.change_Phi_resolution(M=12, N=12)

lambda_regularization = np.append(np.array([0]), np.logspace(-30, 1, 20))

# solve_regularized_surface_current method runs the REGCOIL algorithm
fields, data = solve_regularized_surface_current(
    surface_current_field,  # the surface current field whose geometry and Phi resolution will be used
    eq=eq,  # the Equilibrium object to minimize Bn on the surface of
    current_helicity=(
        1,
        0,
    ),  # pair of integers (M_coil. N_coil), determines topology of contours (almost like  QS helicity),
    #  M_coil is the number of times the coil transits poloidally before closing back on itself
    # and N_coil is the toroidal analog (if M_coil!=0 and N_coil=0, we have modular coils, if both M_coil
    # and N_coil are nonzero, we have helical coils)
    # we pass in an array to perform scan over the regularization parameter (which we call lambda_regularization)
    # to see tradeoff between Bn and current complexity
    eval_grid=LinearGrid(N=20, M=20, NFP=eq.NFP, sym=True),
    source_grid=LinearGrid(N=20, M=20, NFP=eq.NFP),
    lambda_regularization=1e-19,
    # lambda_regularization determines the complexity of the coils, larger values yield simpler coils at the cost of higher Bn error
    vacuum=True,  # this is a vacuum equilibrium, so no need to calculate the Bn contribution from the plasma currents
)

# solve_regularized_surface_current returns a list of FourierCurrentPotentialField objects, one for each lambda, even if a single lambda is passed in
surface_current_field = fields[0]


plot_2d(
    eq.surface,
    "B*n",
    field=surface_current_field,
    cmap="viridis",
);


plot_2d(
    surface_current_field, "Phi", filled=False, levels=20
);  # see how the current potential contours look, these are the coil shapes on the surface


# the to_CoilSet method fits the above contours with SplineXYZCoils
coilset_initial_from_regcoil = surface_current_field.to_CoilSet(
    num_coils=3, stell_sym=True
)
plot_2d(
    eq.surface,
    "B*n",
    field=coilset_initial_from_regcoil,
    cmap="viridis",
)
fig = plot_3d(
    eq.surface,
    "B*n",
    field=coilset_initial_from_regcoil,
    field_grid=coil_grid,
    grid=plot_grid,
)

fig = plot_coils(coilset_initial_from_regcoil, fig=fig)
fig.show()


# we want FourierXYZ to optimize with, so we will fit these with a Fourier series
coilset_initial_from_regcoil = coilset_initial_from_regcoil.to_FourierXYZ(N=10)
plot_2d(
    eq.surface,
    "B*n",
    field=coilset_initial_from_regcoil,
    cmap="viridis",
)
fig = plot_3d(
    eq.surface,
    "B*n",
    field=coilset_initial_from_regcoil,
    field_grid=coil_grid,
    grid=plot_grid,
)

fig = plot_coils(coilset_initial_from_regcoil, fig=fig)
fig.show()


# perform optimization now with this coilset as the initial guess, which has better Bn error than our prior guess

weights = {
    "quadratic flux": 200,
    "coil-coil min dist": 100,
    "plasma-coil min dist": 10,
    "coil curvature": 500,
    "coil length": 2,
}
obj = make_vac_coil_obj(eq, coilset_initial_from_regcoil, weights)
## define our constraints
# we will constrain the current of one coil to avoid the trivial quadratic flux minimizing solution
# of zero coil current
coil_indices_to_fix_current = [False for c in coilset_initial_from_regcoil]
coil_indices_to_fix_current[0] = True
constraints = (
    FixCoilCurrent(coilset_initial_from_regcoil, indices=coil_indices_to_fix_current),
)

# Pick an optimizer. For this problem with only linear constraints we can use a regular least squares method.
# if we used the ToroidalFlux constraint we would need to use a constrained optimization method
optimizer = Optimizer("lsq-exact")

(optimized_coilset_initial_from_regcoil,), _ = optimizer.optimize(
    coilset_initial_from_regcoil,
    objective=obj,
    constraints=constraints,
    maxiter=100,
    verbose=3,
    copy=True,
    ftol=1e-4,
)


normalized_field = compute_average_normalized_field(
    optimized_coilset_initial_from_regcoil, eq, vacuum=True
)
print(f"<Bn> = {normalized_field:.3e}")


## visualize the optimized coilset and the normal field error
# passing in eq.surface avoids the unnecessary plasma field computation (as this is a vacuum eq)
plot_2d(
    eq.surface,
    "B*n",
    field=optimized_coilset_initial_from_regcoil,
    cmap="viridis",
)
fig = plot_3d(
    eq.surface,
    "B*n",
    field=optimized_coilset_initial_from_regcoil,
    field_grid=coil_grid,
    grid=plot_grid,
)

fig = plot_coils(optimized_coilset_initial_from_regcoil, fig=fig)
fig.show()


plot_field_lines(optimized_coilset_initial_from_regcoil, eq);


# ## Finite Beta


# Now let's get a finite beta stellarator, we'll use a W7-X example which has $\beta=2\%$


eq = desc.examples.get("W7-X")


# ### Make Initial Coilset
# 
# Again we will initialize with circular coils centered on the equilibrium axis, and with their normal oriented along the axis tangent vector.
# 


modular_coilset = initialize_modular_coils(eq, num_coils=3, r_over_a=2.5).to_FourierXYZ(
    N=4
)


# To this we'll also add some planar saddle coils. We'll have coils on both the inboard and outboard side, but only 1 on each side per half-period


inner_coils = initialize_saddle_coils(
    eq, num_coils=1, r_over_a=1.0, offset=4.0, position="inner"
)
outer_coils = initialize_saddle_coils(
    eq, num_coils=1, r_over_a=1.0, offset=4.0, position="outer"
)


# Next we'll combine the modular and windowpane coils into a single `MixedCoilSet`. This will allow us to optimize both at the same time.


coilset = MixedCoilSet(inner_coils, outer_coils, modular_coilset)

# visualize the initial coilset
plot_grid = LinearGrid(
    M=40, N=80, NFP=1, endpoint=True
)  # a smaller than usual plot grid to reduce memory of the notebook file

fig = plot_3d(eq, "|B|", grid=plot_grid)
fig = plot_coils(coilset, fig=fig)
fig.show()


# ### Optimizing a `MixedCoilSet` with different types of coils


# We'll use the same grid for all the coils. Alternatively, we could use separate grids
# for the windowpane and modular coils.
coil_grid = LinearGrid(N=50)
# we won't define any plasma grid, but instead use the default grid provided by the objective which is usually sufficient

obj = ObjectiveFunction(
    (
        QuadraticFlux(
            eq,
            field=coilset,
            field_grid=coil_grid,
            vacuum=False,
            weight=200,
        ),
        CoilSetMinDistance(
            coilset,
            bounds=(0.2, np.inf),
            normalize_target=False,  # target is already in normalized units
            weight=20,
            grid=coil_grid,
        ),
        CoilLength(
            coilset,
            bounds=(0, 1.5 * modular_coilset[0].compute("length")["length"]),
            normalize_target=True,  # target in meters
            grid=coil_grid,
        ),
        CoilCurvature(
            coilset,
            # recall that this is signed curvature, in normalized units
            bounds=(-1, 2),
            normalize_target=False,  # target is already in normalized units
            weight=20,
            grid=coil_grid,
        ),
        CoilSetLinkingNumber(coilset, weight=500, grid=LinearGrid(N=25)),
    )
)

# Because the plasma component of the normal field is nonzero, we don't need
# any additional constraints to avoid the trivial solution, but we will include
# constraints to fix the center position of the windowpane coils
constraints = (FixParameters(coilset, params=[{"center": True}, {"center": True}, {}]),)

optimizer = Optimizer("lsq-exact")

(optimized_coilset2,), _ = optimizer.optimize(
    coilset,
    objective=obj,
    constraints=constraints,
    maxiter=50,
    verbose=3,
    copy=True,
)


normalized_field = compute_average_normalized_field(optimized_coilset2, eq)
print(f"<Bn> = {normalized_field:.3e}")


# visualize the optimized coilset and the normal field error
fig = plot_3d(eq, "B*n", field=optimized_coilset2, field_grid=coil_grid, grid=plot_grid)
fig = plot_coils(optimized_coilset2, fig=fig)
fig.show()


# currents in inner saddle coils
optimized_coilset2.coils[0].current


# currents in outer saddle coils
optimized_coilset2.coils[1].current


# current in modular coils
optimized_coilset2.coils[2].current


