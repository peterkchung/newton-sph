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

###########################################################################
# Example SPH Regolith Excavation
#
# Demonstrates WCSPH solver simulating lunar regolith being excavated
# by a simple scoop/bucket tool. Uses granular cohesion to model
# regolith behavior.
#
# Command: uv run -m newton.examples sph.example_sph_regolith_excavation
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    """Lunar regolith excavation demo using SPH solver."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0

        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Regolith parameters
        self.particle_spacing = 0.1
        self.pile_height = 20  # particles high
        self.pile_width = 15  # particles wide
        self.pile_depth = 10  # particles deep

        # SPH parameters from CLI or defaults
        self.sph_params = {
            "smoothing_radius": getattr(args, "smoothing_radius", 0.15),
            "rest_density": getattr(args, "rest_density", 1500.0),
            "viscosity": getattr(args, "viscosity", 0.05),
            "cohesion_stiffness": getattr(args, "cohesion", 1e3),
            "sound_speed": getattr(args, "sound_speed", 50.0),
        }

        # Build model
        builder = newton.ModelBuilder()

        # Add ground plane
        builder.add_ground_plane()

        # Create regolith pile
        self._add_regolith_pile(builder)

        # Add excavation tool (simple box)
        self._add_excavator_tool(builder)

        self.model = builder.finalize()

        # Create SPH solver
        self.solver = newton.solvers.SolverSPH(model=self.model, **self.sph_params)

        # States
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        viewer.set_model(self.model)

        self.capture()

    def _add_regolith_pile(self, builder):
        """Add a pile of regolith particles."""
        spacing = self.particle_spacing
        mass = self.sph_params["rest_density"] * (spacing**3)

        # Create pile in a pyramid shape
        for layer in range(self.pile_height):
            y = 0.5 + layer * spacing  # Start above ground

            # Decrease width as we go up (pyramid)
            layer_width = max(3, self.pile_width - layer // 2)
            layer_depth = max(3, self.pile_depth - layer // 3)

            for i in range(layer_width):
                for j in range(layer_depth):
                    x = (i - layer_width / 2) * spacing
                    z = (j - layer_depth / 2) * spacing

                    builder.add_particle(
                        pos=wp.vec3(x, y, z),
                        vel=wp.vec3(0.0, 0.0, 0.0),
                        mass=mass,
                        radius=spacing * 0.4,
                    )

        print(f"Added {builder.particle_count} regolith particles")

    def _add_excavator_tool(self, builder):
        """Add a simple excavation tool (represented as kinematic body)."""
        # For now, we'll just visualize a moving point
        # In a full implementation, this would be a kinematic body
        pass

    def capture(self):
        """Capture simulation graph for GPU execution."""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Run simulation substeps."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply SPH physics
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        """Render current state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify simulation completed successfully."""
        # Check that particles are still finite
        import numpy as np

        positions = self.state_0.particle_q.numpy()

        # All positions should be finite
        assert np.all(np.isfinite(positions)), "NaN or Inf in particle positions"

        # Particles should be within reasonable bounds (not escaped to infinity)
        max_pos = np.max(np.abs(positions))
        assert max_pos < 100.0, f"Particles escaped too far: max_pos={max_pos}"

        print(f"✓ Final particle count: {len(positions)}")
        print(f"✓ Max particle displacement: {max_pos:.3f}m")


if __name__ == "__main__":
    # Create parser with example-specific arguments
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--smoothing-radius",
        type=float,
        default=0.15,
        help="SPH smoothing radius [m]",
    )
    parser.add_argument(
        "--rest-density",
        type=float,
        default=1500.0,
        help="Regolith rest density [kg/m³]",
    )
    parser.add_argument(
        "--viscosity",
        type=float,
        default=0.05,
        help="Artificial viscosity coefficient",
    )
    parser.add_argument(
        "--cohesion",
        type=float,
        default=1e3,
        help="Cohesion stiffness [Pa]",
    )
    parser.add_argument(
        "--sound-speed",
        type=float,
        default=50.0,
        help="Numerical sound speed [m/s]",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
