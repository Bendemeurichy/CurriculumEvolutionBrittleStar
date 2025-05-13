import time
import warnings
from NEAT.tensorneat.Pipeline import Pipeline
import numpy as np
import os
import jax
from tensorneat.common import State


class NBestPipeline(Pipeline):
    def __init__(
        self,
        *args,
        n_best: int = 10,
        early_stop_distance: float = np.nan,
        early_stop_patience: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_best = n_best
        self.best_genomes: list[tuple] = []
        self.best_fitnesses: list[float] = []
        self.early_stop_distance = early_stop_distance
        self.early_stop_patience = early_stop_patience

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
            worst_top_fitness = min(self.best_fitnesses)

        # Update top N genomes
        for idx, fitness in enumerate(fitnesses):
            if np.isinf(fitness) or fitness <= worst_top_fitness:
                continue

            genome = (pop[0][idx], pop[1][idx])
            self.best_genomes.append(genome)
            self.best_fitnesses.append(fitness)

        # Keep only top N genomes by sorting both lists based on fitness
        if self.best_fitnesses:
            # Sort both lists by fitness (descending)
            sorted_indices = np.argsort(self.best_fitnesses)[::-1]
            self.best_fitnesses = [
                self.best_fitnesses[i] for i in sorted_indices[: self.n_best]
            ]
            self.best_genomes = [
                self.best_genomes[i] for i in sorted_indices[: self.n_best]
            ]

        # For backward compatibility - set best_fitness and best_genome
        if self.best_genomes:
            self.best_fitness = self.best_fitnesses[0]
            self.best_genome = self.best_genomes[0]

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

            # Save each genome with its fitness
            for i, (genome, fitness) in enumerate(
                zip(self.best_genomes, self.best_fitnesses)
            ):
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
        """Copied from the original Pipeline class, but with modifications to return the N best genomes. and implement early stopping.

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """

        print("start compile")
        tic = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The jitted function .* includes a pmap. Using jit-of-pmap can lead to inefficient data movement",
            )
            compiled_step = jax.jit(self.step).lower(state).compile()

        consec = 0

        # count how many generations actually run
        generations = 0

        if self.show_problem_details:
            self.compiled_pop_transform_func = (
                jax.jit(jax.vmap(self.algorithm.transform, in_axes=(None, 0)))
                .lower(state, self.algorithm.ask(state))
                .compile()
            )

        print(
            f"compile finished, cost time: {time.time() - tic:.6f}s",
        )

        for generation in range(self.generation_limit):
            generations += 1
            self.generation_timestamp = time.time()

            # Unpack distances as well
            state, previous_pop, fitnesses, distances = compiled_step(state)

            fitnesses = jax.device_get(fitnesses)
            distances = jax.device_get(distances)

            self.analysis(state, previous_pop, fitnesses)

            # Early stopping: use distances just like fitnesses
            best_distance = np.min(distances)
            if best_distance <= self.early_stop_distance:
                consec += 1
                print(
                    f"Early stopping check: {consec}/{self.early_stop_distance} generations within {self.early_stop_distance} of target (best distance: {best_distance:.4f})"
                )
                if consec >= self.early_stop_patience:
                    print("Early stopping: patience reached.")
                    break
            else:
                consec = 0

            if max(fitnesses) >= self.fitness_target:
                print("Fitness limit reached!")
                break

        if int(state.generation) >= self.generation_limit:
            print("Generation limit reached!")

        if self.is_save:
            best_genome = jax.device_get(self.best_genome)
            with open(os.path.join(self.genome_dir, f"best_genome.npz"), "wb") as f:
                np.savez(
                    f,
                    nodes=best_genome[0],
                    conns=best_genome[1],
                    fitness=self.best_fitness,
                )

        print(f"Took {generations} generations to train")

        return state, self.best_genomes, generations

    def step(self, state: State):
        """Step function for the pipeline. This function is called in a loop to run the NEAT algorithm.

        Args:
            state (State): The current state of the NEAT algorithm.

        Returns:
            State: The updated state after one step.
        """
        # Call the original step function
        state, previous_pop, fitnesses = super().step(state)

        if (
            hasattr(self, "compiled_pop_transform_func")
            and self.compiled_pop_transform_func is not None
        ):
            pop_transformed = self.compiled_pop_transform_func(state, previous_pop)
        else:
            pop_transformed = jax.vmap(self.algorithm.transform, in_axes=(None, 0))(
                state, previous_pop
            )

        self.compiled_compute_distance = jax.jit(
            jax.vmap(
                lambda params: self.problem.compute_distance(
                    state, params, self.algorithm.forward
                )
            )
        )
        distances = self.compiled_compute_distance(pop_transformed)

        return state, previous_pop, fitnesses, distances
