from utils import *
import warnings
warnings.filterwarnings("ignore")

#Parameters:
mask_size = 0.7
mask_value = 10 #inicialización de los nodos desconocidos, valor más o menos arbitrario (se estudirá en detalle más adelante)
message_passing_iterations = 100

visualize_graph = True
save_graph = True
show_graph = False

bg, edges = build_graph_from_csv('data/graph_mix_2019.csv')
visualize(bg, 'pollution')

mask = np.random.choice(a=[True, False], size=(bg.num_nodes()), p=[mask_size, 1-mask_size])
bg_original = bg.ndata['pollution'].clone().detach() 
    
bg.ndata['pollution'][mask] = mask_value
bg.ndata['pollution'][remove] = -1
visualize(bg,'pollution')

MAMP(bg, mask,feature_name = 'pollution', max_iters = message_passing_iterations, plot = True, original_feat = bg_original, visualize_graph = visualize_graph, show = show_graph, save = save_graph)

print('Final error: ', mean_absolute_percentage_error(bg.ndata['pollution'], bg_original))