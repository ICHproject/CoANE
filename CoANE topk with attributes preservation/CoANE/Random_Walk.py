import networkx as nx
import scipy.sparse as sp
import numpy as np
import random

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=True, seed = 0):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    print("seed: ", seed)
    if verbose == True:
        print( 'preprocessing...')

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_matrix(adj)
    orig_num_cc = nx.number_connected_components(g)

    adj_triu = sp.triu(adj) # upper triangular portion of adj matrix
    adj_tuple = sparse_to_tuple(adj_triu) # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0] # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    num_test = int(np.floor(edges.shape[0] * test_frac)) # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac)) # controls how alrge the validation set should be

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples) # initialize train_edges to have all edges
    test_edges = set()
    val_edges = set()

    if verbose == True:
        print( 'generating test/val sets...')

    # Iterate over shuffled edges, add to train/val sets
    np.random.seed(seed)
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print( "WARNING: not enough removable edges to perform full train-test split!")
        print( "Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print( "Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print( 'creating false test edges...')
    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if verbose == True:
        print('creating false val edges...')

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false:
            continue
            
        val_edges_false.add(false_edge)

    if verbose == True:
        print('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, 
            # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false or \
            false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose == True:
        print('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print ('Done with train-test split!')
        print ('')

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, \
        val_edges, val_edges_false, test_edges, test_edges_false

def create_table(subsample_t, sentences):
    
    words = {} 
    nb_sentence = 0
    total = 0. #number of nodes


    for d in sentences:
        nb_sentence += 1
        for w in d:
            if w not in words:
                words[w] = 0
            words[w] += 1
            total += 1

    id2word = {i+1:j for i,j in enumerate(words)} #id to node
    word2id = {j:i for i,j in id2word.items()} #node 2 id
    nb_word = len(words)+1 #number of nodes + padding (id = 0) 

    prob_table = []
    f75_sum = 0

    #sample prob.
    for w in id2word.values():
        prob_table.append(words[w]**(0.75))
        f75_sum += prob_table[-1]

    for i in range(len(prob_table)):
        prob_table[i] /= f75_sum

    #sampling
    subsamples = {i:j/total for i,j in words.items() if j/total > subsample_t}
    subsamples = {i:subsample_t/j+(subsample_t/j)**0.5 for i,j in subsamples.items()}
    subsamples = {word2id[i]:j for i,j in subsamples.items() if j < 1.}
  
    return id2word, word2id, nb_word, subsamples

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

class Graph_walk():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    if self.p==1 and self.q == 1:
                        next = (cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                    else:
                        prev = walk[-2]
                        next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
                            alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length, verbose=True, seed = 0):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        if verbose == True:
            print( 'Walk iteration:')
        for walk_iter in range(num_walks):
            if verbose == True:
                print( str(walk_iter+1), '/', str(num_walks))
            random.seed(seed)
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        Node_probs = []
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            Node_probs.append(normalized_probs)
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if self.p==1 and self.q == 1:
            if is_directed:
                for edge in G.edges():
                    alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            else:
                for edge in G.edges():
                    alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                    alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return Node_probs

def random_walk(network_tuple, args):
    seed = args.seed
    val_frac, test_frac = (args.val_frac, args.test_frac)
    Spliting = args.Spliting

    #random walk parameters
    NUM_WALKS = args.num_walks
    P = args.p 
    Q = args.q 
    WALK_LENGTH = args.walk_length 
    DIRECTED = args.dircted
    subsample_t = args.subsample_rate
    adj, features, dict_NtoC = network_tuple
    edges = {}
    
    
    # Perform train-test split
    if Spliting:#based on node2vec https://github.com/lucashu1/link-prediction
        np.random.seed(seed)
        print('Spliting...')
        if not DIRECTED:
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
                test_edges, test_edges_false = mask_test_edges(adj, test_frac=test_frac, val_frac=val_frac, seed = seed)
        else:
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
                test_edges, test_edges_false = mask_test_edges_directed(adj, test_frac=test_frac, val_frac=val_frac,  prevent_disconnect=False, false_edge_sampling='random' , seed = seed)
  
        g_train = nx.from_scipy_sparse_matrix(adj_train) # new graph object with only non-hidden edges
        name =["train_edges", "train_edges_false", "val_edges", "val_edges_false", "test_edges", "test_edges_false"]
        e_data = [train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false]
        edges = {n:e for n, e in zip(name, e_data)}
    else:
        adj_train = adj
        
    g_train = nx.from_scipy_sparse_matrix(adj_train)
    #graph node list
    gl = list(g_train.nodes())
    gl.sort()    
    
    
    print('Random Walking...')
    # Preprocessing - generate walks (based on https://github.com/aditya-grover/node2vec)
    g_n2v = Graph_walk(g_train, DIRECTED, P, Q) # create node2vec graph instance and P & Q follow the biasd par. p & q from node2vec 
    probwalk = g_n2v.preprocess_transition_probs()
    walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
    walks = [list(map(str, walk)) for walk in walks]

    # map to id and create subsampling prob. table
    sentences = walks
    id2word, word2id, nb_id, subsamples = create_table(subsample_t, sentences)


    #reorder feat by id
    id2word_list = [int(i) for i in id2word.values()]
    feat = features[id2word_list]
    feat = sp.vstack((np.zeros((1,feat.shape[1])), feat))


    #classify (if NUM_WALKS >1) and summary sentences for each repeating generation  

    # sec_cata = {}
    # for s in sentences:
    #     if s[0] in sec_cata:
    #         sec_cata[s[0]].append(s)
    #     else:
    #         sec_cata[s[0]] = [s]

    # source_rep = NUM_WALKS
    # cata_sentences = []
    # i = 0
    # while i < source_rep:
    #     sens = []
    #     for key in sec_cata.keys():
    #         sens.append(sec_cata[key][i])
    #     cata_sentences.append(sens)
    #     i += 1

    cata_sentences = sentences#np.array(cata_sentences)
    
    
    output_k = ['cata_sentences', 'edges', 'gl', 'dict_NtoC', 'id2word', 'word2id', 'subsamples', 'sentences', 'adj_train', 'feat', 'adj', 'nb_id']
    output = [cata_sentences, edges, gl, dict_NtoC, id2word, word2id, subsamples, sentences, adj_train, feat, adj, nb_id]
    PAR = {output_k[i]: output[i] for i in range(len(output_k))}
    
    return PAR
