import fast_pareto

import numpy as np
import matplotlib.pyplot as plt

points = np.random.normal(0, 1, (100, 2))
maximize = np.array([False, True])

is_pareto = fast_pareto.find_pareto_points(points, maximize)

print(points[is_pareto])

plt.scatter(points[:, 0], points[:, 1], c=is_pareto)

plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.show()