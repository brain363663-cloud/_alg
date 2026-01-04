import itertools
import random

# 1. Riemann 
def riemann_nd(func, bounds, steps):
    n = len(bounds)
    
    def recurse(level, point):
        if level == n:
            return func(*point)
        a, b = bounds[level]
        step = (b - a) / steps[level]
        total = 0
        for i in range(steps[level]):
            x = a + (i + 0.5) * step  # midpoint
            total += recurse(level + 1, point + [x])
        return total * step
    
    return recurse(0, [])

# 2. Monte
def monte_carlo_nd(func, bounds, samples):
    n = len(bounds)
    total = 0
    volume = 1
    for a, b in bounds:
        volume *= (b - a)
    
    for _ in range(samples):
        point = [random.uniform(a, b) for (a, b) in bounds]
        total += func(*point)
    
    return total / samples * volume

def test_func(x, y, z):
    return x + y + z  

bounds_3d = [(0, 1), (0, 1), (0, 1)]
steps_3d = [10, 10, 10]
samples = 100000

riemann_result = riemann_nd(test_func, bounds_3d, steps_3d)
montecarlo_result = monte_carlo_nd(test_func, bounds_3d, samples)

print("Riemann 3D result:", riemann_result)
print("Monte Carlo 3D result:", montecarlo_result)
