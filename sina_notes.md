# Stochastic Optimization Notes:

1. `simsopts`'s covariance_function implementation and ours is different. They don't have 2 in the denominator, but the paper does.
2. In `_Coils.compute_magnetic_field`, `position_perturbations` and `tanget_perturbations` must be in XYZ coordinates, rpz is not allowed. Other arguments allow it though.
3. Some children coil objects override `compute_magnetic_field`, should we add perturbations to all of them?
4. `SplineXYZCoil` doesn't use coil tangents?