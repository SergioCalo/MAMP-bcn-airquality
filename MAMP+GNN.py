from utils import *
from models import GCN, SAGE_3, SAGE_5
import warnings
import time
warnings.filterwarnings("ignore")

#Parameters:
mask_value = 10 #inicialización de los nodos desconocidos, valor más o menos arbitrario (se estudirá en detalle más adelante)
gnn_epochs = 200
message_passing_iterations = 100

visualize_graph = False
save_graph = False
show_graph = False

    
mask_sizes = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
loss_list = np.zeros((len(mask_sizes),gnn_epochs))

for i, mask_size in enumerate(mask_sizes):
    bg, edges = build_graph_and_node_features('data/graph_mix_2018.csv')

    mask = np.random.choice(a=[True, False], size=(bg.num_nodes()), p=[mask_size, 1-mask_size])
    bg_original = bg.ndata['pollution'].clone().detach() 

    bg.ndata['pollution'][mask] = mask_value
    bg.ndata['pollution'][remove] = -1
    time0 = time.time()

    MAMP(bg, mask,feature_name = 'pollution', max_iters = message_passing_iterations, plot = False, original_feat = bg_original, visualize_graph = visualize_graph, show = show_graph, save = save_graph)
    time1 = time.time()
    print('time MAMP:', time1- time0)
    print('MAMP MAPE error: ', mean_absolute_percentage_error(bg.ndata['pollution'], bg_original))
    mamp_loss = F.mse_loss(bg.ndata['pollution'], bg_original)
    print('MAMP MSE error: ', mamp_loss)


   # normalized_y = normalize_feature(bg.ndata['y'])
    #normalized_x = normalize_feature(bg.ndata['x'])
    #normalized_noise = normalize_feature(bg.ndata['noise'])
    #normalized_pollution = normalize_feature(bg.ndata['pollution'])

    #node_features = torch.stack([normalized_x, normalized_y, normalized_noise, normalized_pollution], axis=1).squeeze(2).float()
    node_features = bg.ndata['pollution']
    node_labels = bg_original

    train_size = 1
    train_mask = np.random.choice(a=[True, False], size=(bg.num_nodes()), p=[train_size, 1-train_size])
    train_mask = torch.tensor(train_mask)
    valid_mask = torch.tensor(~train_mask)
    n_features = node_features.shape[1]

    #model = GCN(in_feats=n_features, hid_feats=10) #GCN small
    #model = GCN(in_feats=n_features, hid_feats=50) #GCN medium
    #model = GCN(in_feats=n_features, hid_feats=100) #GCN large
    #model = SAGE_3(in_feats=n_features, hid_feats=10) #SAGE3 small
    model = SAGE_3(in_feats=n_features, hid_feats=50) #SAGE3 medium
    #model = SAGE_3(in_feats=n_features, hid_feats=100) #SAGE3 large
    #model = SAGE_5(in_feats=n_features, hid_feats=10) #SAGE5 small
    #model = SAGE_5(in_feats=n_features, hid_feats=50) #SAGE5 medium
    #model = SAGE_5(in_feats=n_features, hid_feats=100) #SAGE5 large




    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(trainable_params, ' trainable parameters')
    print(total_params, ' total parameters')


    opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    best_val = 1000.
    earlystop = 0 #parar entrenamiento si el performance no mejora en x epocas
    time2 = time.time()

    for epoch in range(gnn_epochs):
        model.train()
        # forward propagation by using all nodes
        logits = model(bg, node_features)
        # compute loss
        loss = F.mse_loss(logits[train_mask], node_labels[train_mask])
        # compute validation loss
        val_loss = F.mse_loss(logits[valid_mask], node_labels[valid_mask])
        loss_list[i, epoch] = loss.detach().numpy()
       # print(loss_list)
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch%1000==0:
            print('Epoch: ', epoch)
            print('Train loss: ', loss.item())
            print('Validation loss: ', val_loss.item())
            print('Total graph loss: ', F.mse_loss(logits, bg_original))
        if val_loss.item() < best_val:
            best_val = loss.item()
            earlystop = 0
        else:
            earlystop +=1
        if earlystop == 1000:
            print('earlystop')
            break

        # Save model if necessary.  Omitted in this example.

    time3 = time.time()
    print('time GNN:', time3- time2)


    logits = model(bg, node_features)
    u, v = torch.tensor(edges.FID_x), torch.tensor(edges.FID_y)
    g = dgl.graph((u, v))
    bg_reconstructed = dgl.to_bidirected(g)
    bg_reconstructed.ndata['pollution'] = logits.detach()
    loss = F.mse_loss(logits, node_labels)
    print('mask_size: ', mask_size)
    print('MSE error: ', loss)
    print('MAPE error: ', mean_absolute_percentage_error(bg_reconstructed.ndata['pollution'], bg_original))
    print('------------')
    
    
    
    bg, edges = build_graph_and_node_features('data/graph_mix_2019.csv')
    
    #mask = np.random.choice(a=[True, False], size=(bg.num_nodes()), p=[mask_size, 1-mask_size])
    bg_original = bg.ndata['pollution'].clone().detach() 
    bg.ndata['pollution'][mask] = mask_value
    bg.ndata['pollution'][remove] = -1
    MAMP(bg, mask,feature_name = 'pollution', max_iters = message_passing_iterations, plot = False, original_feat = bg_original, visualize_graph = visualize_graph, show = show_graph, save = save_graph)
    node_features = bg.ndata['pollution']
    
    
    logits = model(bg, node_features)
    u, v = torch.tensor(edges.FID_x), torch.tensor(edges.FID_y)
    g = dgl.graph((u, v))
    bg_reconstructed = dgl.to_bidirected(g)
    bg_reconstructed.ndata['pollution'] = logits.detach()
    loss = F.mse_loss(logits, node_labels)
    print('mask_size: ', mask_size)
    print('MSE error: ', loss)
    print('MAPE error: ', mean_absolute_percentage_error(bg_reconstructed.ndata['pollution'], bg_original))
    print('------------')
    
plt. clf()     
for i, line in enumerate(loss_list):
    plt.plot(line, label = "% of missing nodes: " + str(mask_sizes[i]))
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.show()
