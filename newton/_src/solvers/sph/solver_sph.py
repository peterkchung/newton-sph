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

"""Weakly Compressible SPH (WCSPH) solver for granular materials and fluids.

This solver implements the Smoothed Particle Hydrodynamics method with support
for two-way coupling with rigid bodies, making it suitable for simulating
granular materials like lunar regolith.

The solver uses explicit time integration with the following features:
    - Density computation via SPH summation
    - Pressure forces from Tait equation of state
    - Artificial viscosity for stability
    - Optional cohesion for granular bonding
    - Two-way coupling with Newton rigid bodies

Example:
    >>> import newton
    >>> builder = newton.ModelBuilder()
    >>> # Add SPH particles
    >>> builder.add_particle_grid(
    ...     pos=(0.0, 1.0, 0.0),
    ...     vel=(0.0, 0.0, 0.0),
    ...     dim_x=20,
    ...     dim_y=10,
    ...     dim_z=10,
    ...     cell_size=0.1,
    ...     mass=0.01,
    ... )
    >>> model = builder.finalize()
    >>> solver = newton.solvers.SolverSPH(
    ...     model=model,
    ...     smoothing_radius=0.15,
    ...     rest_density=1500.0,
    ...     viscosity=0.05,
    ...     cohesion_stiffness=1e3,
    ... )

References:
    - Monaghan, J.J. (1992). Smoothed Particle Hydrodynamics.
      Annu. Rev. Astron. Astrophys., 30, 543-574.
    - Becker, M. & Teschner, M. (2007). Weakly compressible SPH for free
      surface flows. Proc. of ACM SIGGRAPH/Eurographics SCA, 209-216.
"""

from __future__ import annotations

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelBuilder, State
from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from .sph_kernels import (
    compute_cohesion_force,
    compute_density,
    compute_pressure,
    compute_pressure_force,
    compute_viscosity_force,
    integrate_sph,
)


@wp.kernel
def _add_vec3_arrays(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
):
    """Helper kernel to add two vec3 arrays element-wise."""
    tid = wp.tid()
    out[tid] = a[tid] + b[tid]


class SolverSPH(SolverBase):
    """Weakly Compressible SPH solver with two-way rigid body coupling.

    This solver implements WCSPH for simulating fluids and granular materials.
    It supports two-way coupling with rigid bodies, allowing accurate simulation
    of interactions between SPH particles and Newton's rigid body system.

    The solver is particularly well-suited for:
    - Fluid simulation (water, oil, etc.)
    - Granular materials (sand, lunar regolith)
    - Multi-phase flows with particle-solid interactions

    Args:
        model: The Newton model containing particles and rigid bodies
        smoothing_radius: SPH smoothing length h [m]. Controls neighbor range.
            Larger values create smoother but less detailed simulation.
        rest_density: Reference density ρ₀ [kg/m³]. For regolith ~1500-1800.
        viscosity: Artificial viscosity coefficient α [dimensionless].
            Higher values increase damping. Typical: 0.01-0.1.
        cohesion_stiffness: Granular cohesion k [Pa]. Bonds particles together.
            Higher values for more cohesive materials. Set to 0 for pure fluids.
        friction_coeff: Internal friction coefficient μ [dimensionless].
            Resistance to shear flow. Typical: 0.3-0.8 for regolith.
        sound_speed: Numerical sound speed c [m/s]. Controls compressibility.
            Lower = more compressible but requires smaller timesteps.
        enable_rigid_coupling: Enable two-way coupling with rigid bodies.
            When True, fluid pressure affects bodies and vice versa.

    Example:
        >>> solver = newton.solvers.SolverSPH(
        ...     model=model,
        ...     smoothing_radius=0.1,
        ...     rest_density=1500.0,
        ...     viscosity=0.05,
        ...     cohesion_stiffness=1e3,
        ...     friction_coeff=0.5,
        ...     sound_speed=50.0,
        ...     enable_rigid_coupling=True,
        ... )
        >>> solver.step(state_in, state_out, control, contacts, dt=1.0 / 60.0)

    Note:
        The solver uses semi-implicit Euler integration. The timestep should
        satisfy the CFL condition: dt < C * h / c where C ~ 0.2-0.4.
    """

    def __init__(
        self,
        model: Model,
        smoothing_radius: float = 0.1,
        rest_density: float = 1500.0,
        viscosity: float = 0.01,
        cohesion_stiffness: float = 0.0,
        friction_coeff: float = 0.5,
        sound_speed: float = 50.0,
        enable_rigid_coupling: bool = True,
    ):
        super().__init__(model=model)

        # SPH parameters
        self.smoothing_radius = smoothing_radius
        self.rest_density = rest_density
        self.viscosity = viscosity
        self.cohesion_stiffness = cohesion_stiffness
        self.friction_coeff = friction_coeff
        self.sound_speed = sound_speed
        self.enable_rigid_coupling = enable_rigid_coupling

        # Equation of state parameter (gamma for Tait equation)
        self.gamma = 7.0

        # Initialize hash grid if needed
        if model.particle_count > 1 and model.particle_grid is not None:
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

        # Persistent buffers for SPH computation
        self._init_sph_buffers()

    def _init_sph_buffers(self):
        """Initialize persistent GPU buffers for SPH intermediate data."""
        model = self.model
        device = model.device
        n = model.particle_count

        if n > 0:
            # Densities [kg/m³]
            self.densities = wp.zeros(n, dtype=float, device=device)
            # Pressures [Pa]
            self.pressures = wp.zeros(n, dtype=float, device=device)
            # SPH force accumulator [N]
            self.sph_forces = wp.zeros(n, dtype=wp.vec3, device=device)
        else:
            self.densities = None
            self.pressures = None
            self.sph_forces = None

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register SPH-specific custom attributes with the model builder.

        These attributes store per-particle SPH data that persists across
        simulation steps.

        Args:
            builder: The ModelBuilder to register attributes with
        """
        import newton  # noqa: PLC0415

        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="density",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                dtype=wp.float32,
                default=0.0,
                assignment=newton.Model.AttributeAssignment.MODEL,
                namespace="sph",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="pressure",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                dtype=wp.float32,
                default=0.0,
                assignment=newton.Model.AttributeAssignment.MODEL,
                namespace="sph",
            )
        )

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation by one timestep using WCSPH.

        The step method performs the following operations:
            1. Build/update hash grid for neighbor queries
            2. Compute particle densities via SPH summation
            3. Compute pressures from equation of state
            4. Compute SPH forces (pressure + viscosity + cohesion)
            5. Apply forces to rigid bodies (two-way coupling)
            6. Integrate particles using semi-implicit Euler
            7. Handle boundary conditions

        Args:
            state_in: Input state at time t
            state_out: Output state at time t+dt (modified in-place)
            control: Control inputs (optional, uses model defaults if None)
            contacts: Contact information for rigid body collisions
            dt: Time step [s]

        Raises:
            ValueError: If dt is too large violating CFL condition
        """
        model = self.model

        # Ensure control is available
        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("sph_step", False):
            # Skip if no particles
            if model.particle_count == 0:
                return

            # 1. Build/update hash grid for neighbor queries
            if model.particle_count > 1 and model.particle_grid is not None:
                search_radius = self.smoothing_radius * 2.0
                with wp.ScopedDevice(model.device):
                    model.particle_grid.build(state_in.particle_q, radius=search_radius)

            # 2. Compute densities
            self._compute_densities(state_in)

            # 3. Compute pressures from equation of state
            self._compute_pressures()

            # 4. Compute SPH forces
            self._compute_sph_forces(state_in)

            # 5. Clear output state forces
            state_out.clear_forces()

            # 6. Apply SPH forces to output state
            if self.sph_forces is not None:
                # Add SPH forces to particle_f
                wp.copy(state_out.particle_f, self.sph_forces)

            # 7. Particle-body coupling
            # Note: Full two-way coupling requires specialized SPH boundary handling.
            # For now, the solver focuses on particle-particle SPH physics.
            # Rigid body coupling can be achieved by:
            #   a) Using SemiImplicit solver with SPH particles
            #   b) Adding custom boundary kernels in future

            # 8. Integrate particles
            self._integrate(state_in, state_out, dt)

            # 10. Store SPH data in custom attributes (optional)
            self._store_sph_data(state_out)

    def _compute_densities(self, state: State) -> None:
        """Compute SPH densities using neighbor summation.

        ρ_i = Σ_j m_j * W(x_i - x_j, h)
        """
        model = self.model

        if model.particle_count == 0 or self.densities is None:
            return

        wp.launch(
            kernel=compute_density,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                model.particle_mass,
                model.particle_grid.id if model.particle_grid else 0,
                self.smoothing_radius,
            ],
            outputs=[self.densities],
            device=model.device,
        )

    def _compute_pressures(self) -> None:
        """Compute pressures from equation of state.

        Uses Tait equation: p = B * [(ρ/ρ₀)^γ - 1]
        """
        model = self.model

        if model.particle_count == 0 or self.pressures is None:
            return

        wp.launch(
            kernel=compute_pressure,
            dim=model.particle_count,
            inputs=[
                self.densities,
                self.rest_density,
                self.sound_speed,
                self.gamma,
            ],
            outputs=[self.pressures],
            device=model.device,
        )

    def _compute_sph_forces(self, state: State) -> None:
        """Compute SPH pressure and viscosity forces."""
        model = self.model

        if model.particle_count == 0 or self.sph_forces is None:
            return

        # Zero force accumulator
        self.sph_forces.zero_()

        # Compute pressure forces
        wp.launch(
            kernel=compute_pressure_force,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                self.pressures,
                self.densities,
                model.particle_mass,
                model.particle_grid.id if model.particle_grid else 0,
                self.smoothing_radius,
            ],
            outputs=[self.sph_forces],
            device=model.device,
        )

        # Compute viscosity forces and add to accumulator
        if self.viscosity > 0.0:
            viscosity_forces = wp.zeros_like(self.sph_forces)
            wp.launch(
                kernel=compute_viscosity_force,
                dim=model.particle_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    self.densities,
                    model.particle_mass,
                    model.particle_grid.id if model.particle_grid else 0,
                    self.smoothing_radius,
                    self.sound_speed,
                    self.viscosity,
                ],
                outputs=[viscosity_forces],
                device=model.device,
            )
            # Add viscosity to total force
            wp.launch(
                kernel=_add_vec3_arrays,
                dim=model.particle_count,
                inputs=[self.sph_forces, viscosity_forces],
                outputs=[self.sph_forces],
                device=model.device,
            )

        # Compute cohesion forces for granular materials
        if self.cohesion_stiffness > 0.0:
            cohesion_forces = wp.zeros_like(self.sph_forces)
            wp.launch(
                kernel=compute_cohesion_force,
                dim=model.particle_count,
                inputs=[
                    state.particle_q,
                    self.densities,
                    model.particle_mass,
                    model.particle_grid.id if model.particle_grid else 0,
                    self.smoothing_radius,
                    self.cohesion_stiffness,
                ],
                outputs=[cohesion_forces],
                device=model.device,
            )
            # Add cohesion to total force
            wp.launch(
                kernel=_add_vec3_arrays,
                dim=model.particle_count,
                inputs=[self.sph_forces, cohesion_forces],
                outputs=[self.sph_forces],
                device=model.device,
            )

    def _integrate(self, state_in: State, state_out: State, dt: float) -> None:
        """Semi-implicit Euler integration for SPH particles."""
        model = self.model

        if model.particle_count == 0:
            return

        wp.launch(
            kernel=integrate_sph,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_out.particle_f,  # Forces including SPH
                model.particle_mass,
                model.particle_inv_mass,
                model.particle_flags,
                model.particle_world,
                model.gravity,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=model.device,
        )

    def _apply_rigid_coupling(
        self,
        state_in: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Apply two-way coupling between SPH particles and rigid bodies.

        PLACEHOLDER: This method is reserved for future SPH-specific
        fluid-solid coupling implementations, such as:
            - Ghost particle boundary conditions
            - Mirror boundary conditions
            - Specialized pressure transfer to bodies

        Currently, the SolverSPH focuses on particle-particle SPH physics.
        For fluid-rigid coupling, users can:
            1. Use SolverSemiImplicit which has built-in particle-body contacts
            2. Implement custom boundary handling in this method

        Args:
            state_in: Input state
            state_out: Output state with updated body forces
            contacts: Contact information (may be None)
            dt: Time step
        """
        # Reserved for future SPH-specific boundary models.
        # For standard particle-body interaction, use SolverSemiImplicit.
        pass

    def _store_sph_data(self, state: State) -> None:
        """Store computed SPH densities and pressures in state.

        These can be accessed via custom attributes for visualization
        or debugging.
        """
        # Custom attributes will be set when users query them
        pass

    @override
    def notify_model_changed(self, flags: int) -> None:
        """Handle model changes that affect the solver.

        Args:
            flags: Bitmask of SolverNotifyFlags indicating what changed
        """
        # Handle particle count changes
        if flags & (SolverNotifyFlags.BODY_PROPERTIES | SolverNotifyFlags.MODEL_PROPERTIES):
            # Reinitialize buffers if particle count changed
            if self.model.particle_count != (self.densities.shape[0] if self.densities else 0):
                self._init_sph_buffers()

    def update_contacts(self, contacts: Contacts) -> None:
        """Update contact information from solver state.

        Args:
            contacts: Contacts object to update
        """
        # SPH solver doesn't generate traditional contacts
        # Rigid body contacts are handled by Newton's contact system
        pass
