import einops
from einops.einops import rearrange, reduce
import torch
from typing import Optional
from day1_tests import test
import utils

# You can test your answer to e.g. question 1 by running "test(your_func, 1)"

def intersect_rays_1d(rays, objs):
    results = []
    for i,x in enumerate(rays):
        ray_sols = t.linalg.solve(t.stack([einops.repeat(rays[i, 1, :2], 'd -> b d', b=objects.size(dim=0)), objects[:, 0, :2] - objects[:, 1, :2]], dim=2), (objects[:, 0, :2] - rays[i, 0, :2]))
        results.append(((ray_sols >= t.Tensor([0, 0])).min(dim=1).values * (ray_sols[:, 1] <= 1)).max())
    return t.stack(results)

