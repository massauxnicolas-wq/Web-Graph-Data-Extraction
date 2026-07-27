import numpy as np
from scipy.spatial import KDTree
import time

# Create a thick line
points = []
for x in range(1000):
    for dy in range(-3, 4):  # thickness 7
        points.append([x, x + dy])
points = np.array(points)

print(f"Total points: {len(points)}")

tree = KDTree(points)
unvisited = set(range(len(points)))

path_x, path_y = [], []
curr = 0
step_radius = 5.0

start = time.time()
while unvisited:
    path_x.append(float(points[curr][0]))
    path_y.append(float(points[curr][1]))
    
    nearby = tree.query_ball_point(points[curr], r=step_radius)
    unvisited.difference_update(nearby)
    
    if not unvisited:
        break
        
    k_search = min(int(4 * step_radius**2) + 100, len(points))
    distances, indices = tree.query(points[curr], k=k_search)
    
    next_curr = None
    for dist, idx in zip(distances, indices):
        if idx in unvisited:
            next_curr = idx
            break
            
    if next_curr is None:
        # fallback
        distances, indices = tree.query(points[curr], k=min(k_search*5, len(points)))
        for dist, idx in zip(distances, indices):
            if idx in unvisited:
                next_curr = idx
                break
        if next_curr is None:
            break
    curr = next_curr

print(f"Time: {time.time() - start:.3f}s, Path len: {len(path_x)}")
print("First 10 path points:", list(zip(path_x[:10], path_y[:10])))
