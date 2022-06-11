from utils import *
import warnings
warnings.filterwarnings("ignore")

#Parameters:
mask_size = 0.9
mask_value = 10 #inicialización de los nodos desconocidos, valor más o menos arbitrario (se estudirá en detalle más adelante)
message_passing_iterations = 30

visualize_graph = False
save_graph = True
show_graph = False

bg, edges = build_graph_from_csv('data/graph_mix_2019.csv')

mask = np.random.choice(a=[True, False], size=(bg.num_nodes()), p=[mask_size, 1-mask_size])
bg_original = bg.ndata['pollution'].clone().detach() 
    
bg.ndata['pollution'][mask] = mask_value
bg.ndata['pollution'][remove] = -1

edge_mse, edge_energy, edge_mape = MAMP_edge(bg, mask, file = 'data/graph_mix_2019.csv', feature_name = 'pollution',weights = 'inverse', max_iters = message_passing_iterations, plot = False, original_feat = bg_original, visualize_graph = visualize_graph, show = show_graph, save = save_graph)

print('Final edge error: ', mean_absolute_percentage_error(bg.ndata['pollution'], bg_original))

#Build the graph again with same mask
bg, edges = build_graph_from_csv('data/graph_mix_2019.csv')
    
bg.ndata['pollution'][mask] = mask_value
bg.ndata['pollution'][remove] = -1

mse, energy, mape = MAMP(bg, mask, feature_name = 'pollution', max_iters = message_passing_iterations, plot = False, original_feat = bg_original, visualize_graph = visualize_graph, show = show_graph, save = save_graph)

print('Final non-edge error: ', mean_absolute_percentage_error(bg.ndata['pollution'], bg_original))

#Build the graph again with same mask
bg, edges = build_graph_from_csv('data/graph_mix_2019.csv')
    
bg.ndata['pollution'][mask] = mask_value
bg.ndata['pollution'][remove] = -1

edge_sqrt_mse, edge_sqrt_energy, edge_sqrt_mape = MAMP_edge(bg, mask, 'data/graph_mix_2019.csv', feature_name = 'pollution', weights = 'inverse_sqrt', max_iters = message_passing_iterations, plot = False, original_feat = bg_original, visualize_graph = visualize_graph, show = show_graph, save = save_graph)

print('Final non-edge error: ', mean_absolute_percentage_error(bg.ndata['pollution'], bg_original))

plt.clf() 
plt.plot(mse, label='No edges')
plt.plot(edge_mse, label='Inverse')
plt.plot(edge_sqrt_mse, label='Inverse sqrt')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.legend()
plt.savefig('Results/edge-vs-noedge-mse.png')
plt.show()
        
plt.clf() 
plt.plot(energy, label='No edges')
plt.plot(edge_energy, label='Inverse')
plt.plot(edge_sqrt_energy, label='Inverse sqrt')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.legend()
plt.savefig('Results/edge-vs-noedge-energy.png')
plt.show()

plt.clf() 
plt.plot(mape, label='No edges')
plt.plot(edge_mape, label='Inverse')
plt.plot(edge_sqrt_mape, label='Inverse sqrt')
plt.xlabel('Iteration')
plt.ylabel('MAPE')
plt.legend()
plt.savefig('Results/edge-vs-noedge-mape.png')
plt.show()