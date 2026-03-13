# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SPH (Smoothed Particle Hydrodynamics) solver for granular materials and fluids.

This module provides a Weakly Compressible SPH (WCSPH) solver implementation
with support for two-way coupling with rigid bodies, making it suitable for
simulating granular materials like lunar regolith.

Example:
    >>> import newton
    >>> model = builder.finalize()
    >>> solver = newton.solvers.SolverSPH(
    ...     model=model,
    ...     smoothing_radius=0.1,
    ...     rest_density=1500.0,
    ...     viscosity=0.01,
    ...     cohesion_stiffness=1e3,
    ... )
    >>> solver.step(state_in, state_out, control, contacts, dt)

References:
    - Monaghan, J.J. (1992). Smoothed Particle Hydrodynamics. Annual Review
      of Astronomy and Astrophysics, 30, 543-574.
"""

from .solver_sph import SolverSPH

__all__ = ["SolverSPH"]
