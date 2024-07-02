import torch
import numpy as np
from pointcept.models.utils.structure import Point

class PointSwarmOptimization:
    def __init__(self, num_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive (particle) weight
        self.c2 = c2  # social (swarm) weight

    def optimize(self, point: Point):
        # Initialize particle positions and velocities
        particles = torch.rand((self.num_particles, *point.coord.shape), device=point.coord.device)
        velocities = torch.rand((self.num_particles, *point.coord.shape), device=point.coord.device) * 0.1

        # Initialize personal best positions and global best position
        personal_best_positions = particles.clone()
        personal_best_scores = self.evaluate(particles, point)
        global_best_position = personal_best_positions[personal_best_scores.argmin()]

        for _ in range(self.max_iter):
            # Update velocities and positions
            r1, r2 = torch.rand(2, *particles.shape, device=point.coord.device)
            velocities = (
                self.w * velocities
                + self.c1 * r1 * (personal_best_positions - particles)
                + self.c2 * r2 * (global_best_position - particles)
            )
            particles += velocities

            # Evaluate new positions
            scores = self.evaluate(particles, point)
            better_scores_mask = scores < personal_best_scores
            personal_best_positions[better_scores_mask] = particles[better_scores_mask]
            personal_best_scores[better_scores_mask] = scores[better_scores_mask]

            # Update global best position
            global_best_position = personal_best_positions[personal_best_scores.argmin()]

        # Update point cloud coordinates with the optimized positions
        point.coord = global_best_position
        return point

    def evaluate(self, particles, point):
        # Example evaluation function: minimize the distance to the original coordinates
        return torch.norm(particles - point.coord, dim=-1).sum(dim=-1)

