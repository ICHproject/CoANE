import numpy as np
import networkx as nx
import scipy
from scipy import sparse
from sklearn.preprocessing import normalize

def Pubmed_read_node(data_dir, dataset):
    with open(data_dir + "/{}/Pubmed-Diabetes.NODE.paper.tab".format(dataset), 'rb') as f:
        raw_idx_features_labels = f.read().decode("utf-8")

    raw_idx_features_labels =  raw_idx_features_labels.split('\n')
    feature_name = {}
    for i, f in enumerate(raw_idx_features_labels[1].split('\t')[1:-1]):
        feature_name[f.split(':')[1]] = i
    nodeid = []
    features = []
    labels = []
    for i, s in enumerate(raw_idx_features_labels):
        ss = s.split('\t')
        if ss[0].isdigit():
            nodeid.append(int(ss[0]))
            labels.append(int(ss[1][-1]))
            s_feature = np.zeros(len(feature_name))
            for f in ss[2:-1]:
                ff = f.split('=')
                s_feature[feature_name[ff[0]]] = float(ff[1])
            features.append(s_feature)
        else:
            print(ss[0])
    nodeid = np.array(nodeid)
    features = np.array(features)
    labels = np.array(labels)    
    return nodeid, features, labels


def Pubmed_read_edges(data_dir, dataset):
    with open(data_dir + "/{}/Pubmed-Diabetes.DIRECTED.cites.tab".format(dataset), 'rb') as f:
        raw_adj = f.read().decode("utf-8")
    edges_unordered = []
    for edge in raw_adj.split('\n')[2:-1]:
        s_edge = edge.split('\t')
        e1 = int(s_edge[1].split(':')[-1])
        e2 = int(s_edge[-1].split(':')[-1])
        edges_unordered.append([e1, e2])
    return edges_unordered

def mat_process(data_dir, dataset):
    data_dict = scipy.io.loadmat(data_dir+"/{}/{}.mat".format(dataset, dataset))
    adj = data_dict['Network']
    features = normalize(data_dict['Attributes'],'l2' )
    m, n = features.shape
    dict_NtoC = data_dict['Label']
    lmin = (dict_NtoC.reshape(-1)).min()
    lmax = (dict_NtoC.reshape(-1)).max()
    dict_NtoC = {i:[int(x[0])+(-lmin)] for i, x in enumerate(dict_NtoC)}
    print("Data with node: ", m, "feature:", n, "classes: ", lmax-lmin+1)
    return (adj, features, dict_NtoC)

def read_dataset(args):
    data_dir, dataset, data = args.data_dir, args.dataset, args.data
    if args.dataset in ['BlogCatalog', 'Flickr', 'Flickr_SDM']:
        network_tuple = mat_process(data_dir, dataset)
        print('Loading Done...')
        return network_tuple
    if dataset=='Pubmed':
        nodeid, features, labels = Pubmed_read_node(data_dir, dataset)
    else:
        #read index-feature-label
        idx_features_labels = np.genfromtxt(data_dir+"/{}/{}.content".format(dataset, data), dtype=np.dtype(str))
        nodeid = idx_features_labels[:, 0]
        features = idx_features_labels[:, 1:-1].astype('float64')
        print(features.shape)
        labels = idx_features_labels[:, -1]

    #map label
    classes = set(labels)
    classes_map = {c:i for i,c in enumerate(classes)}
    #remap id
    idstack = []
    IDstr2int = {} 
    newid = 0
    for id in nodeid:
        idstack.append(newid)
        IDstr2int[str(id)] = newid
        newid += 1
        
    #

    nodeid = np.array(idstack)
    node2label = {int(n):classes_map[l] for n, l in zip(nodeid, labels)}
    print("Data with node: ",features.shape[0], "feature:", features.shape[1], "classes: ", len(classes))

    #read edges
    if dataset=='Pubmed':
        edges_unordered = Pubmed_read_edges(data_dir, dataset)
    else:
        edges_unordered = np.genfromtxt("./datasets/{}/{}.cites".format(dataset, data), dtype=str)
    #remove no-feat node
    edges_intID = []
    no_Feat_node = []
    for edge in edges_unordered:
      e1 = edge[0]
      e2 = edge[1]
      if str(e1) in IDstr2int and str(e2) in IDstr2int:
        edges_intID.append([IDstr2int[str(e1)], IDstr2int[str(e2)]])
      else:
        if str(e1) not in IDstr2int and str(e1) not in no_Feat_node:
          no_Feat_node.append(str(e1))
        if str(e2) not in IDstr2int and str(e2) not in no_Feat_node:
          no_Feat_node.append(str(e2))
          
    edges_intID = np.array(edges_intID)
    g = nx.Graph()
    g.add_edges_from(edges_intID)
    #
    error_node = []
    for node_index, features_series in zip(nodeid, features):
        node_index = int(node_index)
      # Haven't yet seen node (not in edgelist) --> add it now
        if not g.has_node(node_index):
            print(node_index, " does not has edge")
            error_node.append(node_index)
        else:
            g.node[node_index]['features'] = features_series

            if node_index in node2label:
                g.node[node_index]['label'] = [node2label[node_index]]
            else:
                g.node[node_index]['label'] = []
                print(node_index, " does not has label")
    print("#Error node: ", len(error_node), '#node after processing: ', features.shape[0]-len(error_node))
    #check connected
    #if not nx.is_connected(g):
    #    print('number_connected_components: ', nx.number_connected_components(g))

    # Get adjacency matrix in sparse format (sorted by g.nodes())
    adj = nx.adjacency_matrix(g) 
    # Get features matrix (also sorted by g.nodes())
    features = np.zeros_like(features) # num nodes, num features

    #
    dict_NtoC ={}
    for i, node in enumerate(g.nodes()):
        features[i,:] = g.node[node]['features']
        dict_NtoC[i] = g.node[node]['label']

    # Save adj, features in pickle file
    features = sparse.csr_matrix(features)
    network_tuple = (adj, features, dict_NtoC)

    print('Loading Done...')
    return network_tuple #adj, features, dict_NtoC