import numpy as np

def generate_fractal(L, H, N, x=None, i=1):
    """
    Generate a fractal cloud of points

    Parameters
    ----------
    L : int
        Fractal length
    H : int
        Number of levels
    N : int
        Number of points to sample for each point in the previous level
    x : np.ndarray
        Seed positions
    i : int
        Current level (the first level is 1)

    References
    ----------
    The algorithm was implemented following Elmegreen (1997)
    """

    if x is None:
        eta = np.random.random(N)
        x_new = 2 * (eta - 0.5) / L ** i + 0.5
    else:
        eta = np.random.random((N,) + x.shape)
        x_new = x + 2 * (eta - 0.5) / L ** i

    if i < H:
        return generate_fractal(L, H, N, x=x_new.ravel(), i=i+1)
    else:
        return x_new.ravel()

def generate_fractal_grid(L, H, N, dim):
    """
    Generate a fractal cloud of points on a grid

    Parameters
    ----------
    L : int
        Fractal length
    H : int
        Number of levels
    N : int
        Number of points to sample for each point in the previous level
    dim : int
        Dimension of the resulting cube along each side

    Returns
    -------
    grid : np.ndarray
        The output grid with shape (dim, dim, dim)

    References
    ----------
    The algorithm was implemented following Elmegreen (1997)
    """
    x = generate_fractal(L, H, N)
    y = generate_fractal(L, H, N)
    z = generate_fractal(L, H, N)
    values = np.vstack([x, y, z]).transpose()
    return np.histogramdd(values, bins=dim, range=[[0., 1.], [0., 1.], [0., 1.]])[0]
