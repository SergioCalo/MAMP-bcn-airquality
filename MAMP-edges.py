from utils import *
import warnings
warnings.filterwarnings("ignore")

#Parameters:
mask_size = 0.7
mask_value = 10 #inicialización de los nodos desconocidos, valor más o menos arbitrario (se estudirá en detalle más adelante)
message_passing_iterations = 300

visualize_graph = False
save_graph = False
show_graph = False

weights = 'inverse_sqrt'

mask_sizes = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
for mask_size in mask_sizes:
    bg, edges = build_graph_from_csv('data/graph_mix_2019.csv')
    #visualize(bg, 'pollution')

    mask = np.random.choice(a=[True, False], size=(bg.num_nodes()), p=[mask_size, 1-mask_size])
    bg_original = bg.ndata['pollution'].clone().detach() 

    bg.ndata['pollution'][mask] = mask_value
    bg.ndata['pollution'][remove] = -1
    #visualize(bg, 'pollution')

    MAMP_edge(bg, mask, file = 'data/graph_mix_2019.csv', feature_name = 'pollution', weights = weights, max_iters = message_passing_iterations, plot = True, original_feat = bg_original, visualize_graph = visualize_graph, show = show_graph, save = save_graph)

    loss = F.mse_loss(bg_original, bg.ndata['pollution'])
    print('mask_size: ', mask_size)
    print('MSE error: ', loss)
    print('MAPE error: ', mean_absolute_percentage_error(bg.ndata['pollution'], bg_original))
    print('------------')