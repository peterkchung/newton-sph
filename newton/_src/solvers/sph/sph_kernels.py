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

"""Warp kernels for Smoothed Particle Hydrodynamics (SPH) solver.

This module provides GPU-accelerated kernels for WCSPH (Weakly Compressible SPH)
simulation, including:
    - Density summation using smoothing kernels
    - Pressure computation via equation of state
    - Artificial viscosity for stability
    - Cohesion forces for granular materials
    - Two-way coupling with rigid bodies

References:
    - Monaghan, J.J. (1992). Smoothed Particle Hydrodynamics.
      Annual Review of Astronomy and Astrophysics, 30, 543-574.
    - Becker, M. & Teschner, M. (2007). Weakly compressible SPH for free
      surface flows. ACM SIGGRAPH/Eurographics SCA.
"""

from __future__ import annotations

import warp as wp

from ...geometry import ParticleFlags


# SPH smoothing kernels
@wp.func
def kernel_poly6(r: float, h: float) -> float:
    """Poly6 smoothing kernel for SPH.

    The Poly6 kernel is commonly used for density computation due to its
    smooth derivatives and compact support.

    Args:
        r: Distance between particles
        h: Smoothing radius (kernel support)

    Returns:
        Kernel weight value
    """
    if r >= h:
        return 0.0

    h2 = h * h
    h9 = h2 * h2 * h2 * h2 * h
    coeff = 315.0 / (64.0 * wp.pi * h9)
    diff = h2 - r * r
    return coeff * diff * diff * diff


@wp.func
def kernel_spiky_gradient(r: wp.vec3, h: float) -> wp.vec3:
    """Gradient of Spiky kernel for pressure forces.

    The Spiky kernel provides better stability for pressure computation
    compared to Poly6 due to its non-zero gradient at the center.

    Args:
        r: Vector from j to i (r_ij = x_i - x_j)
        h: Smoothing radius

    Returns:
        Gradient of kernel at position r
    """
    r_len = wp.length(r)
    if r_len >= h or r_len < 1e-6:
        return wp.vec3(0.0)

    h6 = h * h * h * h * h * h
    coeff = -45.0 / (wp.pi * h6)
    diff = h - r_len
    factor = coeff * diff * diff / r_len
    return factor * r


@wp.func
def kernel_viscosity_laplacian(r: float, h: float) -> float:
    """Laplacian of viscosity kernel.

    Used for artificial viscosity computation in the momentum equation.

    Args:
        r: Distance between particles
        h: Smoothing radius

    Returns:
        Laplacian of viscosity kernel
    """
    if r >= h:
        return 0.0

    h6 = h * h * h * h * h * h
    coeff = 45.0 / (wp.pi * h6)
    return coeff * (h - r)


# Density computation kernel
@wp.kernel
def compute_density(
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    hash_grid: wp.uint64,
    smoothing_radius: float,
    densities: wp.array(dtype=float),
):
    """Compute SPH density via summation over neighbors.

    ρ_i = Σ_j m_j * W(x_i - x_j, h)

    Args:
        particle_q: Particle positions [m], shape [particle_count]
        particle_mass: Particle masses [kg], shape [particle_count]
        hash_grid: Hash grid ID for neighbor queries
        smoothing_radius: SPH smoothing length h [m]
        densities: Output densities [kg/m³], shape [particle_count]
    """
    tid = wp.tid()
    x_i = particle_q[tid]
    m_i = particle_mass[tid]

    density = m_i * kernel_poly6(0.0, smoothing_radius)

    # Query neighbors
    query = wp.hash_grid_query(hash_grid, x_i, smoothing_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if index == tid:
            continue

        x_j = particle_q[index]
        m_j = particle_mass[index]
        r = wp.length(x_i - x_j)

        density += m_j * kernel_poly6(r, smoothing_radius)

    densities[tid] = density


# Pressure computation kernel
@wp.kernel
def compute_pressure(
    densities: wp.array(dtype=float),
    rest_density: float,
    sound_speed: float,
    gamma: float,
    pressures: wp.array(dtype=float),
):
    """Compute pressure using Tait equation of state.

    p = B * [(ρ/ρ₀)^γ - 1]
    where B = (c²ρ₀)/γ

    Args:
        densities: Computed densities [kg/m³], shape [particle_count]
        rest_density: Reference density ρ₀ [kg/m³]
        sound_speed: Numerical sound speed c [m/s]
        gamma: Stiffness parameter (typically 7 for water)
        pressures: Output pressures [Pa], shape [particle_count]
    """
    tid = wp.tid()
    rho = densities[tid]

    # Tait equation
    B = (sound_speed * sound_speed * rest_density) / gamma
    pressure = B * (wp.pow(rho / rest_density, gamma) - 1.0)

    pressures[tid] = wp.max(pressure, 0.0)  # No negative pressure (tensile instability)


# Pressure force kernel
@wp.kernel
def compute_pressure_force(
    particle_q: wp.array(dtype=wp.vec3),
    pressures: wp.array(dtype=float),
    densities: wp.array(dtype=float),
    particle_mass: wp.array(dtype=float),
    hash_grid: wp.uint64,
    smoothing_radius: float,
    forces: wp.array(dtype=wp.vec3),
):
    """Compute pressure gradient force.

    f_i^p = -Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij

    Args:
        particle_q: Particle positions [m]
        pressures: Pressures [Pa]
        densities: Densities [kg/m³]
        particle_mass: Masses [kg]
        hash_grid: Hash grid ID
        smoothing_radius: Smoothing length h
        forces: Output forces [N]
    """
    tid = wp.tid()
    x_i = particle_q[tid]
    p_i = pressures[tid]
    rho_i = densities[tid]
    m_i = particle_mass[tid]

    force = wp.vec3(0.0)

    # Query neighbors
    query = wp.hash_grid_query(hash_grid, x_i, smoothing_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if index == tid:
            continue

        x_j = particle_q[index]
        p_j = pressures[index]
        rho_j = densities[index]
        m_j = particle_mass[index]

        r_ij = x_i - x_j
        grad_w = kernel_spiky_gradient(r_ij, smoothing_radius)

        # Symmetric pressure term
        pressure_term = p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)
        force -= m_j * pressure_term * grad_w

    forces[tid] = force


# Artificial viscosity kernel
@wp.kernel
def compute_viscosity_force(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    particle_mass: wp.array(dtype=float),
    hash_grid: wp.uint64,
    smoothing_radius: float,
    sound_speed: float,
    alpha: float,
    forces: wp.array(dtype=wp.vec3),
):
    """Compute Monaghan-type artificial viscosity.

    Provides stability for shock and high-velocity flows by adding
    dissipative forces when particles approach each other.

    The viscosity term is:
        Π_ij = (-α c̄ μ_ij + β μ_ij²) / ρ̄
    where:
        μ_ij = (h v_ij · r_ij) / (|r_ij|² + ε)
        c̄ = average sound speed
        ρ̄ = average density

    Args:
        particle_q: Positions [m]
        particle_qd: Velocities [m/s]
        densities: Densities [kg/m³]
        particle_mass: Masses [kg]
        hash_grid: Hash grid ID
        smoothing_radius: Smoothing length h
        sound_speed: Numerical sound speed c [m/s]
        alpha: Viscosity coefficient (0.01-0.1 typical)
        forces: Output forces [N]
    """
    tid = wp.tid()
    x_i = particle_q[tid]
    v_i = particle_qd[tid]
    rho_i = densities[tid]

    force = wp.vec3(0.0)

    query = wp.hash_grid_query(hash_grid, x_i, smoothing_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if index == tid:
            continue

        x_j = particle_q[index]
        v_j = particle_qd[index]
        rho_j = densities[index]
        m_j = particle_mass[index]

        r_ij = x_i - x_j
        v_ij = v_i - v_j
        r_len = wp.length(r_ij)

        if r_len < smoothing_radius and r_len > 1e-6:
            # Compute viscosity term μ_ij
            dot_vr = wp.dot(v_ij, r_ij)
            epsilon = 0.01 * smoothing_radius * smoothing_radius
            mu = smoothing_radius * dot_vr / (r_len * r_len + epsilon)

            if dot_vr < 0:  # Approaching particles
                avg_c = sound_speed  # Use solver sound speed
                avg_rho = 0.5 * (rho_i + rho_j)

                # Monaghan artificial viscosity with β = α/2 for stability
                beta = alpha * 0.5
                viscosity_term = (-alpha * avg_c * mu + beta * mu * mu) / avg_rho

                # Viscosity force is symmetric
                grad_w = kernel_spiky_gradient(r_ij, smoothing_radius)
                force += m_j * viscosity_term * grad_w

    forces[tid] = force

    forces[tid] = force


# Integration kernel
@wp.kernel
def integrate_sph(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    """Semi-implicit Euler integration for SPH particles.

    Similar to base solver integration but operates on accumulated SPH forces.

    Args:
        particle_q: Current positions [m]
        particle_qd: Current velocities [m/s]
        particle_f: Accumulated forces [N] (SPH forces + external)
        particle_mass: Masses [kg]
        particle_inv_mass: Inverse masses [1/kg]
        particle_flags: Particle flags (active/inactive)
        particle_world: World indices for gravity lookup
        gravity: Gravity vectors per world [m/s²]
        dt: Time step [s]
        v_max: Maximum velocity limit [m/s]
        x_new: Output positions [m]
        v_new: Output velocities [m/s]
    """
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        x_new[tid] = particle_q[tid]
        v_new[tid] = particle_qd[tid]
        return

    x0 = particle_q[tid]
    v0 = particle_qd[tid]
    f0 = particle_f[tid]

    inv_mass = particle_inv_mass[tid]
    world_idx = particle_world[tid]
    world_g = gravity[wp.max(world_idx, 0)]

    # Semi-implicit Euler: v1 = v0 + a*dt, x1 = x0 + v1*dt
    v1 = v0 + (f0 * inv_mass + world_g * wp.step(-inv_mass)) * dt

    # Velocity limit for stability
    v1_mag = wp.length(v1)
    if v1_mag > v_max:
        v1 *= v_max / v1_mag

    x1 = x0 + v1 * dt

    x_new[tid] = x1
    v_new[tid] = v1
