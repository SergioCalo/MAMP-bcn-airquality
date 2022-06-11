from utils import *


bg = build_grid(size = 7)
bg.ndata['x'][10:13] =   3
visualize(bg)

print(homophily_mean(bg))
print(homophily(bg))