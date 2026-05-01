import dgl
import torch

# Build same graph both ways
src, dst = [0, 1, 2], [1, 2, 0]

# Old way
g_old = dgl.DGLGraph()
g_old.add_nodes(3)
g_old.add_edges(src, dst)
g_old.ndata['h'] = torch.ones(3, 4)

# New way
g_new = dgl.graph((src, dst), num_nodes=3)
g_new.ndata['h'] = torch.ones(3, 4)

# Test update_all
import dgl.function as fn
g_old.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'out'))
g_new.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'out'))

print(g_old.ndata['out'])
print(g_new.ndata['out'])
print(torch.allclose(g_old.ndata['out'], g_new.ndata['out']))