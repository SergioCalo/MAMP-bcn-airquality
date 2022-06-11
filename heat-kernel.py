from utils import *
import pygsp as pg
import warnings
warnings.filterwarnings("ignore")

#Parameters:
mask_size = 0.99
mask_value = 10 #inicialización de los nodos desconocidos, valor más o menos arbitrario (se estudirá en detalle más adelante)
n_iter = 6000

bg, edges = build_graph_from_csv('data/graph_mix_2019.csv')
nx_graph = bg.to_networkx().to_undirected()
A = nx.adjacency_matrix(nx_graph)
coor = list(positions.values())
coor = np.array([list(x) for x in coor])
graph = pg.graphs.Graph(A.todense())
graph.set_coordinates(coor[:9466])
graph.compute_fourier_basis()

mask_sizes = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
for mask_size in mask_sizes:
    signal = np.array(bg.ndata['pollution'].T[0])
    s = np.zeros(graph.N)

    mask = np.random.choice(a=[True, False], size=(bg.num_nodes()), p=[mask_size, 1-mask_size])
    bg_original = bg.ndata['pollution'].clone().detach() 
    s[~mask] = signal[~mask]
    s[mask] = 10

    losses = []
    iters = []
    times = np.arange(n_iter)

    for i in times:
        g = pg.filters.Heat(graph, 0.5)
        s = g.filter(s)
        s[~mask] = signal[~mask]
        if i%100 == 0:
            loss = F.mse_loss(torch.tensor(s), torch.tensor(signal))
            #print(loss)
            losses.append(loss)
            iters.append(i)

    loss = F.mse_loss(torch.tensor(s), torch.tensor(signal))
    print('mask_size: ', mask_size)
    print('MSE error: ', loss)
    print('MAPE error: ', mean_absolute_percentage_error(torch.tensor(signal), torch.tensor(s)))
    print('------------')

