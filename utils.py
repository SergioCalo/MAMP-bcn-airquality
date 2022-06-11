import dgl
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import copy
import pickle
import pandas as pd
import torch.nn.functional as F
import math
def homophily(graph):
    h = 0
    for node in range(graph.num_nodes()):
        Ni = graph.edges()[1][graph.edges()[0] == node]
        if len(Ni)>0:
            fm = min(graph.ndata['x'].T[0][Ni])
            fi = graph.ndata['x'].T[0][node]
            fk = graph.ndata['x'].T[0][Ni]
            hi = sum((fi - fm ) / (fk - fm + 0.01)) / len(Ni)
            h += hi
        
    h = h/graph.num_nodes()
    return h

def homophily_mean(graph, feature = 'x'):
    h = 0
    for node in range(graph.num_nodes()):
        Ni = graph.edges()[1][graph.edges()[0] == node]
        if len(Ni)>0:
            fi = graph.ndata[feature].T[0][node]
            fk = graph.ndata[feature].T[0][Ni]
            hi = sum( (fi - fk)**2 ) / len(Ni)
            h += hi
        
    h = h/graph.num_nodes()
    return h

#Nodos sobrantes, limpiamos los datos
remove = [2295, 2296, 3642, 3669, 7765, 7766, 7767, 7768, 7769, 7770, 7771, 7772, 7773, 7774, 7775, 7776, 7930, 7931, 7932, 8014, 8015, 8708, 8709, 8762, 8763, 8925, 8926, 8927, 8928, 9079, 9080, 9081, 9092, 9093, 9094, 9095, 9096, 9105, 9106, 9107, 9108, 9109, 9110, 9111, 9112, 9113, 9114, 9115, 9116, 9117, 9118, 9119, 9120, 9121, 9122, 9123, 9124, 9125, 9126, 9212, 9213, 9271, 9272, 9273, 9274, 9275, 9276, 9278, 9279, 9280, 9281, 9321, 9325, 9326, 9327, 9328, 9329]

with open('data/positions.pkl', 'rb') as f:
    positions = pickle.load(f)

def visualize(graph, feature_name = 'x', show = True, save = False, epoch = False):
    nx_graph = graph.to_networkx().to_undirected()
    nx_graph.remove_nodes_from(remove)
    graph.ndata[feature_name][remove] = -1
    color = graph.ndata[feature_name][graph.ndata[feature_name] != -1]
    plt.figure(figsize=[15,7])
    ec = nx.draw_networkx_edges(nx_graph, pos=positions, alpha=0.7)
    nc = nx.draw_networkx_nodes(nx_graph, pos=positions, node_size=2, node_color=color,  vmin=10, vmax=25, cmap=plt.cm.jet)
    ticks = np.linspace(graph.ndata[feature_name].min(), graph.ndata[feature_name].max(), 5, endpoint=True)
    plt.colorbar(nc, ticks=ticks)
    if save == True:
        fig = plt.gcf()
        fig.savefig('Results/Frames/BCN' + str(epoch) + '.jpg', dpi=100)
    if show == True:
        plt.show()

def weight(row, alpha):
    return alpha * (row.TOTAL_D +  row.Rang * row.LONGITUD) + (1 - alpha) * row.LONGITUD


def mean(x):
    if len(x) > 0:
        return  sum(x)/len(x)
    else:
        return -1


def build_graph_from_csv(file):

    edges = pd.read_csv(file)
    edges = edges.drop_duplicates(subset=['FID_x','FID_y']).reset_index()    

    alpha = 1
    edges['weights'] = edges.apply (lambda row: weight(row, alpha), axis=1)
    node_features = []

    for node in range(max(edges.FID_x)+1):
        result = edges[(edges['FID_x'] == node) | (edges['FID_y'] == node)].Rang
        if len(result) > 0:
            node_features.append([node, sum(result)/len(result) ])
        else:
            node_features.append([node, -1])

    node_features_pd = pd.DataFrame(node_features, columns=['nodes', 'features'])
        
    u, v = torch.tensor(edges.FID_x), torch.tensor(edges.FID_y)
    g = dgl.graph((u, v))
    g.edata['length'] = torch.tensor(edges.LONGITUD)
    bg = dgl.add_reverse_edges(g, copy_edata=True)
    bg.ndata['pollution'] = torch.zeros(bg.num_nodes(), 1)

    for i, node in node_features_pd.iterrows():
        bg.ndata['pollution'][node] = node['features']

    return bg, edges

    
def build_graph_and_node_features(file):
    edges = pd.read_csv(file)
    nodes = pd.read_csv('data/BCN_GrafVial_CSV/BCN_GrafVial_Nodes_ETRS89_CSV.csv',sep=';', encoding='latin-1')
    edges = edges.drop_duplicates(subset=['FID_x','FID_y']).reset_index()
    node_features = []
    for node in range(9465+1):
        
        pollution = edges[(edges['FID_x'] == node) | (edges['FID_y'] == node)].Rang
        noise = edges[(edges['FID_x'] == node) | (edges['FID_y'] == node)].TOTAL_D
        node_features.append([node, mean(pollution), mean(noise), nodes['Coord_X'].iloc[node], nodes['Coord_Y'].iloc[node] ])

    u, v = torch.tensor(edges.FID_x), torch.tensor(edges.FID_y)
    g = dgl.graph((u, v))
    bg = dgl.to_bidirected(g)   
    bg.ndata['x'] = torch.tensor(nodes['Coord_X'][:9466]/100000.).unsqueeze(0).T
    bg.ndata['y'] = torch.tensor(nodes['Coord_Y'][:9466]/1000000.).unsqueeze(0).T
    bg.ndata['pollution'] = torch.zeros(bg.num_nodes(), 1)
    bg.ndata['noise'] = torch.zeros(bg.num_nodes(), 1)

    node_features_pd = pd.DataFrame(node_features, columns = ['nodes', 'pollution', 'noise', 'Coord_X', 'Coord_Y' ])

    for i, node in node_features_pd.iterrows():
        bg.ndata['pollution'][i] = node['pollution']
        bg.ndata['noise'][i] = node['noise']


    return bg, edges

def aggregate(h_j):
    if len(h_j) != 0:
        h_i = sum(h_j)/len(h_j)
        return h_i
    else:
        return 0
    
def weighted_aggregate(h_j, weights):
    if len(h_j) != 0:
        h_j = h_j * weights
        h_i = sum(h_j) / sum(weights)
        return h_i
    else:
        return 0
    
def combine(h_i, m_i):
    if h_i == 0:
        return m_i
    else:
        return (h_i + m_i) / 2.
    
def MAMP(graph, mask, feature_name = 'pollution', max_iters = 1,  early_stop = 5e-3, plot = False, original_feat = False, visualize_graph = False, show = False, save = False):
    original = copy.deepcopy(graph)
    mse = []
    homo = []
    mape_list = []
    for epoch in range(max_iters):
        for node in graph.nodes()[list(graph.ndata[feature_name] != -1)]:
            nb = graph.edges()[1][graph.edges()[0] == node]
            nb_feat = graph.ndata[feature_name][nb]
            m = aggregate(nb_feat[nb_feat!=-1])
            h_i = combine(graph.ndata[feature_name][node], m)
            graph.ndata[feature_name][node] = h_i
            graph.ndata[feature_name][remove] = -1
        graph.ndata[feature_name][~mask] = original.ndata[feature_name][~mask]
        loss = F.mse_loss(original_feat, graph.ndata[feature_name])
        mape = mean_absolute_percentage_error(graph.ndata[feature_name], original_feat)
        mse.append(loss)
        homo.append(homophily_mean(graph, feature_name))
        mape_list.append(mape)
        if visualize_graph == True:
            visualize(graph, feature_name = feature_name, show = show, save=save, epoch = epoch)
            
        if epoch > 5:
            if (mse[-2] - loss) < early_stop:
                print('Convergence found in', epoch ,'iterations')
                break

    if plot == True:
        plt.clf() 
        plt.plot(mse)
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.savefig('Results/mse.png')
      #  plt.show()
        
        plt.clf() 
        plt.plot(homo)
        plt.xlabel('Iteration')
        plt.ylabel('energy')
        plt.savefig('Results/energy.png')
     #   plt.show()

        plt.clf() 
        plt.plot(mape_list)
        plt.xlabel('Iteration')
        plt.ylabel('MAPE')
        plt.savefig('Results/mape.png')
     #   plt.show()
    else:
        return mse, homo, mape_list
        
def MAMP_edge(graph, mask, file, feature_name = 'x', weights = 'inverse', max_iters = 1,  early_stop = 5e-3, plot = False, original_feat = False, visualize_graph = False, show = False, save = False):
    original = copy.deepcopy(graph)
    mse = []
    homo = []
    mape_list = []
    edges = pd.read_csv(file)
    for epoch in range(max_iters):
        for node in graph.nodes()[list(graph.ndata[feature_name] != -1)]:
            nbs = graph.edges()[1][graph.edges()[0] == node]
            lengths = []
            for nb in nbs:
                if len(edges[(edges['FID_x'] == node.item())  & (edges['FID_y'] == nb.item())].LONGITUD) > 0:
                    lengths.append(edges[(edges['FID_x'] == node.item())  & (edges['FID_y'] == nb.item())].LONGITUD.values[0])
                else:
                    lengths.append(edges[(edges['FID_x'] == nb.item()) & (edges['FID_y'] == node.item())].LONGITUD.values[0] )
            nb_feat = graph.ndata[feature_name][nbs]
            if weights == 'inverse':
                m = weighted_aggregate(nb_feat[nb_feat!=-1], torch.tensor(lengths)**-1)
                
            if weights == 'inverse_sqrt':
                m = weighted_aggregate(nb_feat[nb_feat!=-1], torch.sqrt(torch.tensor(lengths))**-1)   
                
            if weights == 'sqrt':
                m = weighted_aggregate(nb_feat[nb_feat!=-1], torch.sqrt(torch.tensor(lengths))) 
            h_i = combine(graph.ndata[feature_name][node], m)
            graph.ndata[feature_name][node] = h_i
            graph.ndata[feature_name][remove] = -1

        graph.ndata[feature_name][~mask] = original.ndata[feature_name][~mask]
        loss = F.mse_loss(original_feat, graph.ndata[feature_name])
        mape = mean_absolute_percentage_error(graph.ndata[feature_name], original_feat)
        mse.append(loss)
        homo.append(homophily_mean(graph, feature_name))
        mape_list.append(mape)
        if visualize_graph == True:
            visualize(graph, feature_name = feature_name, show = show, save=save, epoch = epoch)
            
        if epoch > 5:
            if (mse[-2] - loss) < early_stop:
                print('Convergence found in', epoch ,'iterations')
                break

    if plot == True:
        plt.clf() 
        plt.plot(mse)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig('Results/edge-mse.png')
        plt.show()
        
        plt.clf() 
        plt.plot(homo)
        plt.xlabel('Iteration')
        plt.ylabel('energy')
        plt.savefig('Results/edge-energy.png')
        plt.show()

        plt.clf() 
        plt.plot(mape_list)
        plt.xlabel('Iteration')
        plt.ylabel('MAPE')
        plt.savefig('Results/edge-mape.png')
        plt.show()
    else:
        return mse, homo, mape_list
        
def build_grid(size = 6, features = 'None'):
    X, Y = size, size
    edges_in = []
    edges_out = []

        #Horizontales
    for y in range(Y):
        y=y*Y
        for x in range(X):
            if (x+y)%X !=0:
                edges_out.append((x+y))
                edges_in.append((x+y)-1)


        #Verticales
    for x in range(X-1):
        x=x*X
        for y in range(Y):
          #  if (x+y)%Y !=0:
                edges_out.append((x+y))
                edges_in.append((x+y)+X)   
                
    positions = dict()
    for x in range(X):
        x=x*X
        for y in range(Y):
            positions[x+y] = (x/X, y)
            
    u, v = torch.tensor(edges_in), torch.tensor(edges_out)
    g = dgl.graph((u, v), num_nodes=X*Y)
    bg = dgl.to_bidirected(g)
    bg.ndata['x'] = torch.ones(bg.num_nodes(), 1)
    if features == 'arange':
        bg.ndata['x'] = torch.arange(start=0, end=bg.num_nodes(), step=1).unsqueeze(0).T

    
    G = dgl.to_networkx(bg)
    ec = nx.draw_networkx_edges(G, pos=positions, alpha=0.7)
    nc = nx.draw_networkx_nodes(G, pos=positions, node_color=bg.ndata['x'], cmap=plt.cm.jet)
    plt.colorbar(nc)
    plt.show()
    return bg

def mean_absolute_percentage_error(x, true_label):
    p_e = abs(x - true_label) / true_label
    return  100 * mean(p_e)
    
    
def normalize_feature(feature):
    mu = sum(feature)/len(feature)
    std = torch.std(feature, unbiased=False)
    normalized = (feature - mu)/std
    return normalized

