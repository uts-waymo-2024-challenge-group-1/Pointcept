import torch
import numpy as np
from pointcept.models.utils.structure import Point
from scipy.stats import entropy

class EnhancedPointSwarmOptimization:
    def __init__(self, num_particles=30, max_iter=100, w=0.5, c1=2, c2=2):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive (particle) weight
        self.c2 = c2  # social (swarm) weight

    def optimize(self, target_point: Point, source_point: Point = None):
        # Determine device
        device = target_point.coord.device

        # Handle batch processing
        batch_size = len(target_point.offset) - 1 if 'offset' in target_point else 1
        
        # Adaptive parameter tuning
        self.w = torch.tensor([self.w] * batch_size, device=device)
        self.c1 = torch.tensor([self.c1] * batch_size, device=device)
        self.c2 = torch.tensor([self.c2] * batch_size, device=device)

        if source_point:
            # Initialize particle positions and velocities based on source_point
            particles = torch.rand((batch_size, self.num_particles, 3), device=device)
            particles = self.utilize_serialization(particles=particles, target_point=target_point)
            velocities = torch.zeros((batch_size, self.num_particles, 3), device=device)
            personal_best_positions = particles.clone()
            personal_best_scores = self.evaluate(particles, source_point, target_point)
        else:
            # Initialize particle positions and velocities randomly within the space defined by target_point
            min_bound, max_bound = target_point.coord.min(dim=0).values, target_point.coord.max(dim=0).values
            particles = torch.rand((batch_size, self.num_particles, 3), device=device) * (max_bound - min_bound) + min_bound
            particles = self.utilize_serialization(particles=particles, target_point=target_point)
            velocities = torch.zeros((batch_size, self.num_particles, 3), device=device)
            personal_best_positions = particles.clone()
            personal_best_scores = self.evaluate_void(particles, target_point)

        global_best_position = personal_best_positions[torch.arange(batch_size), personal_best_scores.argmin(dim=1)]

        # Utilize grid-based operations for coarse alignment
        if 'grid_coord' in target_point:
            particles = self.coarse_align(particles, target_point.grid_coord, target_point.grid_size)

        for iter in range(self.max_iter):
            r1, r2 = torch.rand(2, batch_size, self.num_particles, 3, device=device)
            velocities = (
                self.w.view(batch_size, 1, 1) * velocities
                + self.c1.view(batch_size, 1, 1) * r1 * (personal_best_positions - particles)
                + self.c2.view(batch_size, 1, 1) * r2 * (global_best_position.unsqueeze(1) - particles)
            )
            particles += velocities

            if source_point:
                scores = self.evaluate(particles, source_point, target_point)
            else:
                scores = self.evaluate_void(particles, target_point)

            better_scores_mask = scores < personal_best_scores
            personal_best_positions[better_scores_mask] = particles[better_scores_mask]
            personal_best_scores[better_scores_mask] = scores[better_scores_mask]
            global_best_position = personal_best_positions[torch.arange(batch_size), personal_best_scores.argmin(dim=1)]

            # Adaptive parameter tuning
            self.update_parameters(iter)

        if source_point:
            R = self.calculate_rotation_matrix(global_best_position)
            rotated_source = torch.matmul(source_point.coord, R.transpose(-1, -2))
            return rotated_source
        else:
            return self.generate_new_points(global_best_position, target_point)

    def coarse_align(self, particles, grid_coord, grid_size):
        # Implement coarse alignment using grid coordinates
        grid_particles = torch.div(particles, grid_size, rounding_mode='floor')
        return grid_particles * grid_size

    def update_parameters(self):
        # Implement adaptive parameter tuning
        self.w = self.w * 0.99  # Decrease inertia weight
        self.c1 = self.c1 * 0.99  # Decrease cognitive weight
        self.c2 = self.c2 * 1.01  # Increase social weight

    def evaluate(self, particles, source_point, target_point):
        scores = []
        for batch_particles in particles:
            batch_scores = []
            for particle in batch_particles:
                R = self.calculate_rotation_matrix(particle)
                rotated_source = torch.matmul(source_point.coord, R.T)
                
                source_entropy = self.calculate_entropy(rotated_source)
                target_entropy = self.calculate_entropy(target_point.coord)
                entropy_diff = torch.abs(source_entropy - target_entropy)
                mse = torch.mean((rotated_source - target_point.coord) ** 2)
                score = entropy_diff + mse 
                batch_scores.append(score)
            scores.append(torch.stack(batch_scores))
        return torch.stack(scores)

    def evaluate_void(self, particles, target_point):
        scores = []
        for batch_particles in particles:
            batch_scores = []
            for particle in batch_particles:
                target_entropy = self.calculate_entropy(target_point.coord)
                entropy_diff = torch.abs(self.calculate_entropy(particle.unsqueeze(0)) - target_entropy)
                mse = torch.mean((particle - target_point.coord.mean(dim=0)) ** 2)
            
                score = entropy_diff + mse 
                batch_scores.append(score)
            scores.append(torch.stack(batch_scores))
        return torch.stack(scores)


    def calculate_rotation_matrix(self, angles):
        rx, ry, rz = angles.unbind(-1)
        Rx = torch.stack([
            torch.stack([torch.ones_like(rx), torch.zeros_like(rx), torch.zeros_like(rx)], dim=-1),
            torch.stack([torch.zeros_like(rx), torch.cos(rx), -torch.sin(rx)], dim=-1),
            torch.stack([torch.zeros_like(rx), torch.sin(rx), torch.cos(rx)], dim=-1)
        ], dim=-2)
        Ry = torch.stack([
            torch.stack([torch.cos(ry), torch.zeros_like(ry), torch.sin(ry)], dim=-1),
            torch.stack([torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry)], dim=-1),
            torch.stack([-torch.sin(ry), torch.zeros_like(ry), torch.cos(ry)], dim=-1)
        ], dim=-2)
        Rz = torch.stack([
            torch.stack([torch.cos(rz), -torch.sin(rz), torch.zeros_like(rz)], dim=-1),
            torch.stack([torch.sin(rz), torch.cos(rz), torch.zeros_like(rz)], dim=-1),
            torch.stack([torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz)], dim=-1)
        ], dim=-2)
        R = torch.matmul(Rz, torch.matmul(Ry, Rx))
        return R

    def calculate_entropy(self, point_cloud):
        projections = [point_cloud[:, i] for i in range(3)]
        entropies = []
        for proj in projections:
            proj_normalized = (proj - proj.min()) / (proj.max() - proj.min())
            proj_discretized = (proj_normalized * 1000).int()
            unique, counts = torch.unique(proj_discretized, return_counts=True)
            prob = counts.float() / len(proj_discretized)
            ent = entropy(prob.cpu().numpy())
            entropies.append(ent)
        return sum(entropies)

    def generate_new_points(self, best_position):
        batch_size = best_position.shape[0]
        new_points = []
        for i in range(batch_size):
            batch_new_points = []
            for _ in range(self.num_particles):
                perturbation = torch.rand(3, device=best_position.device) * 0.1  # Small random perturbation
                new_point = best_position[i] + perturbation
                batch_new_points.append(new_point)
            new_points.append(torch.stack(batch_new_points))
        return torch.stack(new_points)
    
    def utilize_serialization(self, particles, target_point):
        if 'serialized_code' in target_point:
            # Use serialization information to guide the search
            serialized_code = target_point.serialized_code    
            # Sort particles based on their proximity to serialized points
            sorted_particles = []
            for batch_particles in particles:
                distances = torch.cdist(batch_particles, serialized_code)
                sorted_indices = torch.argsort(distances, dim=1)
                sorted_batch_particles = batch_particles[torch.arange(batch_particles.shape[0]).unsqueeze(1), sorted_indices]
                sorted_particles.append(sorted_batch_particles)
            
            return torch.stack(sorted_particles)
        else:
            return particles
        
    def fine_tune(self, particles, grid_coord, grid_size):
        # Implement fine-tuning using grid coordinates
        grid_particles = torch.div(particles, grid_size, rounding_mode='floor')
        nearest_grid_points = torch.argmin(torch.cdist(grid_particles, grid_coord), dim=2)
        fine_tuned_particles = grid_coord[nearest_grid_points] * grid_size + grid_size / 2
        return fine_tuned_particles