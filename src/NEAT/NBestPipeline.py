import time
from tensorneat.pipeline import Pipeline
import numpy as np
import os
import jax
from tensorneat.common import State


class NBestPipeline(Pipeline):
    def __init__(self, *args, n_best: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_best = n_best
        self.best_genomes: list[tuple] = []

    def analysis(self, state: State, pop, fitnesses):
        """Modified analysis method to store top N genomes."""
        generation = int(state.generation)

        valid_fitnesses = fitnesses[~np.isinf(fitnesses)]
        # avoid there is no valid fitness in the whole population
        if len(valid_fitnesses) == 0:
            max_f, min_f, mean_f, std_f = ["NaN"] * 4
        else:
            max_f, min_f, mean_f, std_f = (
                max(valid_fitnesses),
                min(valid_fitnesses),
                np.mean(valid_fitnesses),
                np.std(valid_fitnesses),
            )

        new_timestamp = time.time()
        cost_time = new_timestamp - self.generation_timestamp

        # Optimization: Only consider genomes that might make it into the top N
        worst_top_fitness = float("-inf")
        if len(self.best_genomes) >= self.n_best:
            worst_top_fitness = self.best_genomes[-1][0]

        # Update top N genomes
        for idx, fitness in enumerate(fitnesses):
            if np.isinf(fitness) or fitness <= worst_top_fitness:
                continue

            genome = (pop[0][idx], pop[1][idx])
            self.best_genomes.append((fitness, genome))

        # Keep only top N genomes by sorting and slicing
        self.best_genomes.sort(
            key=lambda x: x[0], reverse=True
        )  # Sort by fitness (descending)
        self.best_genomes = self.best_genomes[: self.n_best]  # Keep only top N

        # For backward compatibility - set best_fitness and best_genome
        if self.best_genomes:
            self.best_fitness = self.best_genomes[0][0]
            self.best_genome = self.best_genomes[0][1]

        if self.is_save:
            # save best genome of this generation
            max_idx = np.argmax(fitnesses)
            best_genome = jax.device_get((pop[0][max_idx], pop[1][max_idx]))
            file_name = os.path.join(self.genome_dir, f"{generation}.npz")
            with open(file_name, "wb") as f:
                np.savez(
                    f,
                    nodes=best_genome[0],
                    conns=best_genome[1],
                    fitness=fitnesses[max_idx],
                )

            # Save all top N genomes
            top_dir = os.path.join(self.save_dir, "top_genomes")
            if not os.path.exists(top_dir):
                os.makedirs(top_dir)

            # Our list is already sorted
            for i, (fitness, genome) in enumerate(self.best_genomes):
                genome = jax.device_get(genome)
                file_name = os.path.join(
                    top_dir, f"top_{i+1}_fitness_{fitness:.4f}.npz"
                )
                with open(file_name, "wb") as f:
                    np.savez(
                        f,
                        nodes=genome[0],
                        conns=genome[1],
                        fitness=fitness,
                    )

            # append log
            with open(os.path.join(self.save_dir, "log.txt"), "a") as f:
                f.write(f"{generation},{max_f},{min_f},{mean_f},{std_f},{cost_time}\n")

        print(
            f"Generation: {generation}, Cost time: {cost_time * 1000:.2f}ms\n",
            f"\tfitness: valid cnt: {len(valid_fitnesses)}, max: {max_f:.4f}, min: {min_f:.4f}, mean: {mean_f:.4f}, std: {std_f:.4f}\n",
        )

        self.algorithm.show_details(state, fitnesses)

        if self.show_problem_details:
            pop_transformed = self.compiled_pop_transform_func(state, pop)
            self.problem.show_details(
                state, state.randkey, self.algorithm.forward, pop_transformed
            )

    def auto_run(self, state):
        # Run the parent's auto_run implementation but discard its return value
        state, _ = super().auto_run(state)

        # Return state and all best genomes instead of just the best one
        return state, self.best_genomes
