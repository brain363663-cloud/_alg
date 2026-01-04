def compute_gradients(X, y, params):
    # params = [w0, w1]
    N = X.shape[0]
    preds = params[0] + params[1] * X
    error = preds - y
    grad_w0 = 2.0 * np.mean(error)
    grad_w1 = 2.0 * np.mean(error * X)
    return np.array([grad_w0, grad_w1])

def gradient_descent(X, y, init_params=None, lr=0.1, iterations=1000, verbose=False):
    params = np.zeros(2) if init_params is None else np.array(init_params, dtype=float)
    for it in range(iterations):
        grad = compute_gradients(X, y, params)
        params -= lr * grad
        if verbose and (it % (iterations // 10 + 1) == 0):
            print(f"iter {it}, loss {mse_loss(X,y,params):.6f}")
    return params, mse_loss(X, y, params)

def sgd(X, y, init_params=None, lr=0.01, epochs=50, batch_size=1, seed=None, verbose=False):
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    params = np.zeros(2) if init_params is None else np.array(init_params, dtype=float)
    for epoch in range(epochs):
        perm = rng.permutation(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            Xb, yb = X[idx], y[idx]
            # gradients on mini-batch
            preds = params[0] + params[1] * Xb
            error = preds - yb
            grad_w0 = 2.0 * np.mean(error)
            grad_w1 = 2.0 * np.mean(error * Xb)
            grad = np.array([grad_w0, grad_w1])
            params -= lr * grad
        if verbose and (epoch % (epochs // 5 + 1) == 0):
            print(f"epoch {epoch}, loss {mse_loss(X,y,params):.6f}")
    return params, mse_loss(X, y, params)
