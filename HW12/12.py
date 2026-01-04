import numpy as np
import math

np.set_printoptions(precision=5, suppress=True)

def cross_entropy(p, q):
    eps = 1e-15
    q = np.clip(q, eps, 1.0)
    return -np.sum(p * np.log2(q))

def entropy(p):
    return cross_entropy(p, p)

def hill_climb_ce(p, q_init, steps=20000, step_size=0.5):
    q = q_init.copy()
    best_loss = cross_entropy(p, q)

    for t in range(steps):
        noise = np.random.randn(len(q))
        q_try = q + step_size * noise

        q_try = np.abs(q_try) + 1e-12
        q_try = q_try / np.sum(q_try)

        loss = cross_entropy(p, q_try)

        if loss < best_loss:
            q = q_try
            best_loss = loss

        if t % 1000 == 0 and t > 0:
            step_size *= 0.8
            print(f"{t:05d} | CE={best_loss:.6f} | q={q}")

    return q, best_loss

if __name__ == "__main__":
    p = np.array([0.5, 0.25, 0.25])
    q0 = np.array([1/3, 1/3, 1/3])

    print("Target p :", p)
    print("Entropy(p):", entropy(p), "\n")

    q_opt, ce_opt = hill_climb_ce(p, q0)

    print("\nFinal Result")
    print("q_opt :", q_opt)
    print("p     :", p)
    print("Cross Entropy:", ce_opt)

    error = np.sum(np.abs(q_opt - p))
    print("L1 error:", error)

    if error < 1e-2:
        print("\nConverged: q ≈ p (minimum cross-entropy)")
    else:
        print("\nNot converged")
