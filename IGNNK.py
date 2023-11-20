from utils import *
from basic_structure import IGNNK
import warnings
warnings.filterwarnings("ignore")

#Parameters:
mask_size = 0.99
mask_value = 10 
gnn_epochs = 1000

PATH = 'models/IGNNK_bcn_250iter_2023-11-13 16:30:16.pth'
h = 1
z = 100
K = 1
model = IGNNK(h, z, K)
model.load_state_dict(torch.load(PATH))
model.eval()

A, X = load_bcn_data()

 
mask = np.random.choice(a=[True, False], size=(X.shape[0]), p=[mask_size, 1-mask_size])

Mf_inputs = X.detach().clone() / 25.
Mf_inputs[mask] = 0.5
Mf_inputs=Mf_inputs.T.unsqueeze(0)



A_q = torch.from_numpy((calculate_random_walk_matrix(A).T).astype('float32'))
A_h = torch.from_numpy((calculate_random_walk_matrix(A.T).T).astype('float32'))    
    
    
X_res = model(Mf_inputs, A_q, A_h)  #Obtain the reconstruction
print('solution', X_res* 25.)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(trainable_params, ' trainable parameters')
print(total_params, ' total parameters')


bg, edges = build_graph_and_node_features('data/graph_mix_2019.csv')
bg_original = bg.ndata['pollution'].clone().detach() 
u, v = torch.tensor(edges.FID_x), torch.tensor(edges.FID_y)
g = dgl.graph((u, v))
bg_reconstructed = dgl.to_bidirected(g)

bg_reconstructed.ndata['pollution'] = X_res[0,0,:].detach() * 25.
visualize(bg_reconstructed, 'pollution')

print('Final error: ', mean_absolute_percentage_error(bg_reconstructed.ndata['pollution'].unsqueeze(1), bg_original))
print('MSE: ', F.mse_loss(bg_reconstructed.ndata['pollution'], bg_original[:,0]))
