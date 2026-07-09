import numpy as np
import math
import matplotlib.pyplot as plt

def sphere(x):
    """Sphere test function (minimum at 0)"""
    return np.sum(x**2)

def initialize_population(dim, lb, ub, pop_size):
    """Initialize population and calculate initial fitness"""
    population = lb + (ub - lb) * np.random.rand(pop_size, dim)
    fitness = np.array([sphere(ind) for ind in population])
    best_idx = np.argmin(fitness)
    return population, fitness, population[best_idx].copy(), fitness[best_idx]

def misboa_optimize(objective_func, dim, lb, ub, max_iter, pop_size=30):
    """Main optimization process with new_pop for permutations"""
    # === Feedback Regulation Mechanism (Eq. 18) ===
    #Parameter of PID (Proportional–integral–derivative)
    Kp = 0.5  # Proportional gain
    Ki = 0.1  # Integral gain
    Kd = 0.1  # Derivative gain
    epsilon = 1e-100  # Numerical stability threshold

    pop, fitness, best_sol, best_fit = initialize_population(dim, lb, ub, pop_size)
    e_k_1 = best_sol - pop
    e_k_2 = e_k_1.copy()

    convergence = []

    for t in range(max_iter):
        # === Feedback-based movement update (Eq. 18) ===
        e_k = best_sol - pop #Eq . 15
        r1, r2, r3 = np.random.rand(3)
        delta_u = (Kp * r1 * (e_k - e_k_1) + 
                   Ki * r2 * e_k + 
                   Kd * r3 * (e_k - 2 * e_k_1 + e_k_2)) #Eq . 17

        e_k_2, e_k_1 = e_k_1.copy(), e_k.copy()

        # === Compute A and H(t) for Eq. 18 ===
        r4, r5 = np.random.rand(2)
        lambda_val = r4 * math.cos(t / max_iter)
        rho = (math.log(max_iter - t + 2) / math.log(max_iter)) ** 2
        L = 0.01 * np.random.standard_cauchy((pop_size, dim))  # Lévy flight
        H = (math.cos(1 - t / max_iter) + rho * r5 * L) * e_k

        # === Update position using Eq. 18 into new_pop ===
        new_pop = pop + lambda_val * delta_u + (1 - lambda_val) * H # x(t+1) = x(t)+lamda*delta(t)+1(1-lamda)
        new_pop = np.clip(new_pop, lb, ub)

        # === Recalculate fitness for new_pop ===
        new_fitness = np.array([objective_func(ind) for ind in new_pop])
        improved = new_fitness < fitness
        pop[improved] = new_pop[improved]
        fitness[improved] = new_fitness[improved]

        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fit:
            best_sol = pop[current_best_idx].copy()
            best_fit = fitness[current_best_idx]

        # === Hunting Strategy ===
        new_pop = pop.copy()  # Reset new_pop for hunting phase
        if t < max_iter / 3: # Searching for prey
            for i in range(pop_size):
                r1, r2 = np.random.choice([j for j in range(pop_size) if j != i], 2, False)
                R1 = np.random.rand(dim)
                new_pop[i] = pop[i] + (pop[r1] - pop[r2]) * R1  #Eq 2
                new_pop[i] = np.clip(new_pop[i], lb, ub)
        elif max_iter / 3 <= t < 2 * max_iter / 3:
            RB = np.random.randn(pop_size, dim) # Consuming prey   #Eq 4
            for i in range(pop_size):
                new_pop[i] = best_sol + np.exp((t/max_iter)**4) * (RB[i] - 0.5) * (best_sol - pop[i]) #Eq 5-6
                new_pop[i] = np.clip(new_pop[i], lb, ub)
        else: # Attacking prey
            # === Golden Sinusoidal Guidance Strategy (Eq. 19) for attacking phase ===
            s1 = np.random.uniform(0, 2 * math.pi, pop_size)
            s2 = np.random.uniform(0, math.pi, pop_size)
            tau = (1 - math.sqrt(5)) / 2
            theta1 = -math.pi + 2 * math.pi * (1 - tau)
            theta2 = -math.pi + 2 * math.pi * tau

            for i in range(pop_size):
                sin_s1 = abs(math.sin(s1[i]))
                term1 = pop[i] * sin_s1
                term2 = s2[i] * sin_s1 * (theta1 * best_sol - theta2 * best_sol)
                new_pop[i] = term1 + term2
                new_pop[i] = np.clip(new_pop[i], lb, ub)

        # === Update population based on hunting strategy ===
        new_fitness = np.array([objective_func(ind) for ind in new_pop])
        improved = new_fitness < fitness
        pop[improved] = new_pop[improved]
        fitness[improved] = new_fitness[improved]

        # === Update best solution after hunting ===
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fit:
            best_sol = pop[current_best_idx].copy()
            best_fit = fitness[current_best_idx]

        # === Escape Strategy ===
        new_pop = pop.copy()  # Reset new_pop for escape phase
        if np.random.rand() > 0.5: # Camouflage based on environment
            RB = np.random.randn(pop_size, dim)
            for i in range(pop_size):
                new_pop[i] = best_sol + (2*RB[i] - 1) * (1 - t/max_iter)**2 * pop[i]  #Eq 11
                new_pop[i] = np.clip(new_pop[i], lb, ub)
        else:
            for i in range(pop_size): 
                similarities = [] 
                for j in range(pop_size):
                    if i != j:
                        A = pop[i] - best_sol #Eq 21 A = Xi(t) − Xbest(t) 
                        B = pop[j] - best_sol  #     B = Xi(t) − Xbest(t)
                        norm_A = np.linalg.norm(A)
                        norm_B = np.linalg.norm(B)
                        if norm_A < epsilon or norm_B < epsilon:
                            continue
                        sim = np.dot(A, B) / (norm_A * norm_B) #Eq 22
                        similarities.append((j, sim))
                if similarities: #Eq 20
                    j, _ = min(similarities, key=lambda x: x[1]) # Select least similar agent (X_b)
                    R = np.random.rand(dim) # r_6 (random coefficient)
                    K = np.random.randint(1, 3) # Scaling factor (algorithm-specific)
                    new_pop[i] = best_sol + R * (pop[j] - K * pop[i]) #Eq 23 
                    new_pop[i] = np.clip(new_pop[i], lb, ub)

        # === Update population based on escape strategy ===
        new_fitness = np.array([objective_func(ind) for ind in new_pop])
        improved = new_fitness < fitness
        pop[improved] = new_pop[improved]
        fitness[improved] = new_fitness[improved]

        # === Update best solution after escape ===
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fit:
            best_sol = pop[current_best_idx].copy()
            best_fit = fitness[current_best_idx]

        convergence.append(best_fit)
        if best_fit < 1e-100:
            print(f"Numerical convergence at iteration {t}")
            break

        if t % 100 == 0:
            print(f"Iteration {t}: Best Fitness = {best_fit:.4e}")

    best_sol = np.where(np.abs(best_sol) < 1e-50, 0, best_sol)
    if np.all(np.abs(best_sol) < 1e-10):
        best_sol = np.zeros(dim)
        best_fit = 0.0

    return best_sol, best_fit, convergence

if __name__ == "__main__":
    dim = 30
    lb = -100
    ub = 100
    max_iter = 500

    solution, fitness, convergence = misboa_optimize(sphere, dim, lb, ub, max_iter)

    print("\nOptimization Results:")
    print(f"Best solution: {solution}")
    print(f"Best fitness: {fitness:.6e}")

    plt.figure(figsize=(10, 6))
    plt.semilogy(convergence)
    plt.title("MISBOA Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.grid(True)
    plt.show()