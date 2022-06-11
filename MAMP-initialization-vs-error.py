from utils import *
import warnings
warnings.filterwarnings("ignore")

#Parameters:
mask_size = 0.7
mask_values = [0, 5, 10, 12.8628, 20] 
message_passing_iterations = 50

visualize_graph = False
save_graph = True
show_graph = True

for mask_value in mask_values:
    bg, edges = build_graph_from_csv('data/graph_mix_2019.csv')
    mask = np.random.choice(a=[True, False], size=(bg.num_nodes()), p=[mask_size, 1-mask_size])
    bg_original = bg.ndata['pollution'].clone().detach() 

    bg.ndata['pollution'][mask] = mask_value
    bg.ndata['pollution'][remove] = -1
    mse, energy, mape = MAMP(bg, mask,feature_name = 'pollution', max_iters = message_passing_iterations, plot = False, original_feat = bg_original, visualize_graph = visualize_graph, show = show_graph, save = save_graph)
    loss = F.mse_loss(bg_original, bg.ndata['pollution'])
    
    plt.plot(mse, label='Initialization: ' + str(mask_value))

plt.xlabel('Iteration')
plt.ylabel('MSE')   
plt.legend()
plt.savefig('Results/initialization-vs-error.png')
plt.show()

