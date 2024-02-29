import numpy as np

def find_pareto_points(
    points: np.ndarray,
    maximize: np.ndarray | None = None
) -> np.ndarray:
    """Find Pareto frontier of a given array of points.

    Args:
        points: An M x N array of points to evaluate for Pareto-optimality.
        maximize (Optional): A length-N array indicating whether to maximize
            each component. If not provided, all components will be maximized.
    
    Returns:
        A length-M boolean array indicating whether each point is on the Pareto
        front.

    Raises:
        ValueError: Length of maximize array must match the number of
        components in points.
    """
    if maximize is None:
        maximize = np.ones(points.shape[1], dtype=bool)
    else:
        if len(maximize) != points.shape[1]:
            raise ValueError("Length of maximize array must match the number of components in points.")

    opt_points = np.where(maximize[np.newaxis, :], points, -points)

    pairwise_differences = opt_points[:, np.newaxis, :] - opt_points[np.newaxis, :, :]

    is_point_component_suboptimal = (pairwise_differences < 0)
    is_point_suboptimal = np.all(is_point_component_suboptimal, axis=-1)
    suboptimal = np.any(is_point_suboptimal, axis=1)

    on_pareto = np.logical_not(suboptimal)

    return on_pareto
