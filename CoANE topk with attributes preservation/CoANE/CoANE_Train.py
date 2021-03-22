# train
import torch
import torch.nn as nn
import numpy as np
from CoANE.CoANE import *
from tqdm import tqdm
from CoANE.CoANE_Evaluation import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import scipy.sparse as sp

import pickle
import time

# warnings.filterwarnings('ignore')
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')


torch.manual_seed(0)
np.random.seed(0)

def win_b_extract(index_b, winlist, tarlist, n_c_, n_c_s , rand = True):
    n_b = len(index_b)
    v_win_list_b = []
    labels_b = []
    labels_b_set = []
    pos_win = []
    for v in winlist[index_b].tolist():
        v_win_list_b += v
        # pos_win.append(v[np.random.randint(len(v))])
        pos_win.append(np.array(v)[np.random.choice(len(v), 2, replace=False if len(v)>1 else True)].tolist())
    for v in tarlist[index_b].tolist():
        labels_b += v
        labels_b_set += [v[0]]
        
    # generate sampling index
    n_c_b = np.pad(np.cumsum(n_c_[index_b])[:-1], [(1, 0), ], mode='constant', constant_values=0) #starting index for each node
    if rand:
        index_w = [np.random.randint(n_c_[index_b][k], size=n_c_s[index_b][k])+n_c_b[k] for k in range(n_b)] #randomly select windows for training
    else:
        index_w = [np.arange(n_c_[index_b][k])+n_c_b[k] for k in range(n_b)]

    index_w = np.array(index_w, dtype = 'object')
    index_w = np.concatenate(index_w, 0).astype(int)
    # sampling windows
    v_win_list_b = np.array(v_win_list_b, dtype = 'object')[index_w]
    labels_b = np.array(labels_b)[index_w]

    return v_win_list_b, labels_b, labels_b_set, pos_win

def Train_batch_processing(id_node, V_BATCH_SIZE):
    np.random.shuffle(id_node)
    mbatch = []
    num_mbatch = max(int(len(id_node)/V_BATCH_SIZE), 1)
    mbatch.append(id_node[:V_BATCH_SIZE])
    i = 0
    if (V_BATCH_SIZE*(i+1))*2 < len(id_node):
        for i in range(1, num_mbatch):
            mbatch.append(id_node[(i*(V_BATCH_SIZE)):(V_BATCH_SIZE*(i+1))])
    if (V_BATCH_SIZE*(i+1)) < len(id_node):
        num_mbatch+=1
        mbatch.append(id_node[(V_BATCH_SIZE*(i+1))::])
    return mbatch, num_mbatch

def window_initial(Context, args):
    # context number
    n_c_ = np.array(Context['mulcount_list'][0])
    assert np.min(n_c_[1:])>0
    n_c_s = n_c_
    # collect context for each node
    v_win_list = [] 
    labels = []
    for cts in Context['mulreformfn'][0]:
        win = []
        v_list = []
        if len(cts):
            for ct in  cts:
                win+=[ct]
                v_list.append(ct[args.window_hsize])#target ID
        v_win_list.append(win)
        labels.append(v_list)
    v_win_list = np.array(v_win_list, dtype = 'object')
    labels = np.array(labels, dtype = 'object')
    return n_c_, n_c_s, v_win_list, labels

def Train(PAR, Context, args):
    eps = 1e-10
    if 'data':
        verbose = args.verbose
        seed = args.seed
        nb_node = PAR['nb_id']
        nb_filter = args.emb_dim 

        win_len = 2*args.window_hsize+1
        max_win_count = Context["Max_win_count"]

        num_epochs = args.num_epochs
        V_BATCH_SIZE = min(args.num_Vbatch, nb_node)
        R_SIZE =  len(Context["mulreformfn"])
        device = args.device

        #INPUT initial
        id_node = list(range(1, nb_node))
        num_mbatch = max(int(nb_node/V_BATCH_SIZE), 1)
        #Input initial
        X, Y = [], []

        #controller
        r2 = torch.from_numpy(np.array(Context["mulnegFrea"][0])).type(torch.FloatTensor).to(device)
        if 'add DANE pars':
            t_feat = PAR['feat'].todense()
            t_feat = torch.from_numpy(t_feat).float().to(device)
        feat_dim = t_feat.shape[1]
        x_callfeat = torch.arange(nb_node).long().to(device)
        d_feat = t_feat

    print('Initial......')
    for i in range(R_SIZE):
        #sparse window_initial
        n_c_, n_c_s, v_win_list, labels = window_initial(Context, args)
        
        Context["mulcount_list"][i][0]+=1
        #neg  samples
        x_negSample = torch.from_numpy(np.array(Context["mulnegFre"][i].todense()).astype(int)).type(torch.LongTensor).to(device) 
        #co-occurance matrix
        y_D = torch.from_numpy( Context["mulDmatrix"][i].todense()+Context["mulComatrix_1hop"][i].todense()).type(torch.FloatTensor).to(device)
        D_ = ( Context["mulDmatrix"][i]+Context["mulComatrix_1hop"][i])
        #

        if 1 and 'D_sparse':
            D_top_v = []
            D_top_i = []
            n_topk = np.min([Context['Max_win_count'], (D_>0).sum(-1).max()])
            for d_, i_ in zip(np.array_split(D_.data, D_.indptr[1:-1]),np.array_split(D_.indices, D_.indptr[1:-1])):
                n_choose = np.min([n_topk, len(d_)])
                i_sort = np.argpartition(-d_, n_choose-1)[:n_choose]

                D_top_v.append(d_[i_sort].tolist()+[0]*(n_topk-n_choose))
                D_top_i.append(i_[i_sort].tolist()+[0]*(n_topk-n_choose))
            D_top_i = torch.LongTensor(D_top_i).to(device)
            D_top_v = torch.FloatTensor(D_top_v).to(device)
        
        #input data
        x = [v_win_list, x_negSample, labels]
        y = [y_D]

    Context= None
    #MODEL initial
    if 'MODEL initial':
        model = CoANE(feat_dim, nb_filter, win_len, t_feat).to(device)
        model.device = device
        if 0 and 'emb stack':
            mem_emb = nn.Embedding(nb_node, nb_filter).to(device)
            mem_emb.weight.requires_grad = False
        else:
            mem_feat_avg = torch.zeros(nb_node, nb_filter).to(device)
            nn.init.uniform_(mem_feat_avg, -1.0, 1.0)

        #Optimizer
        optimizer = torch.optim.Adam(list(model.parameters()), weight_decay=args.decay)

        #Train model
        print('Training...')
        #if controller is not given
        if 'contoller' not in args:
            r2 = r2[0].detach().cpu().item()
        else:
            r2 = args.contoller
    t_d = 0
    for epoch in range(num_epochs):
        t_s = time.time()
        model.train()

        ###Split V_Batch
        np.random.seed(seed)
        np.random.shuffle(id_node)
        mbatch, num_mbatch = Train_batch_processing(id_node, V_BATCH_SIZE)
        ###
        loss_r = []
        for index_b in mbatch:
            # batching input
            v_win_list_b, labels_b, labels_b_set, pos_win = win_b_extract(index_b, x[0], x[2], n_c_, n_c_s, rand = False)
            #Part 1--------------------------------------
            #Generate emb
            _,  feat_avg = model([v_win_list_b, labels_b, labels_b_set])

            #Update memory emb
            if 'Topk':                
                mem_feat_avg[index_b] = feat_avg.detach()
                gather_pos_emb = mem_feat_avg[D_top_i[index_b].view(-1)].view(len(index_b), n_topk, -1)
                gather_pos_emb = torch.transpose(gather_pos_emb, 1,2)
                D_top_v_b = D_top_v[index_b]

            #Part 2--------------------------------------
            #Loss
            loss_pos = ((( torch.bmm(feat_avg.unsqueeze(1), gather_pos_emb).squeeze().sigmoid()+eps).log()*-D_top_v_b).sum()) #topk
            rec_sim = torch.matmul(feat_avg, torch.transpose(feat_avg, 0, 1)).sigmoid()
            loss_neg = (-(1-rec_sim+1e-10).log()).sum()*1e-3
            if args.c_f:
                loss_mse = model.MSE(model.forward_f(feat_avg), d_feat[index_b])*args.c_f # original
            else:
                loss_mse = torch.zeros_like(loss_neg)
            loss = loss_pos + loss_neg + loss_mse 
            loss_r.append([loss.item(), loss_pos.item(), loss_neg.item(), loss_mse.item()])
            
            #Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        t_e = time.time()
        if verbose:
            loss_r = np.array(loss_r).sum(0)
            print(" Epoch [%d/%d], \n Avg. Loss:  {:%.2f} -p {:%.2f} -n {:%.2f} -m {:%.2f}" % ( epoch+1, \
                                        num_epochs, loss_r[0]/len(id_node),\
             loss_r[1]/len(id_node), loss_r[2]/len(id_node), loss_r[3]/len(id_node) ) )
            if (epoch+1)%5==0 and (epoch+1)<num_epochs:
                t_d += t_e-t_s
                PAR = eval_emb(model, x[0], x[2], nb_filter, V_BATCH_SIZE, PAR, n_c_)
                LP = evaluation(PAR)

        torch.cuda.empty_cache()
            
    #Renew
    print("Renewing embeddings...")
    PAR = eval_emb(model, x[0], x[2], nb_filter, V_BATCH_SIZE, PAR, n_c_)
    print('Training Done!')
    return PAR, model

def eval_emb(model, x_win, x_tar, nb_filter, V_BATCH_SIZE, PAR, n_c_):
    n_c_s = n_c_
    nb_node = len(x_win)
    embeddings = np.zeros((nb_node, nb_filter))
    m = V_BATCH_SIZE
    model.eval()
    index_ = list(range(nb_node))
    
    with torch.no_grad():
        for l in range(1,int(np.ceil(embeddings.shape[0]/m))+1):
            index_b = index_[((l-1)*m+1):(l*m+1)]
            v_win_list_b, labels_b, labels_b_set, _ = win_b_extract(index_b, x_win, x_tar, n_c_, n_c_s, rand = False)
            #Part 1--------------------------------------
            #Generate emb
            embeddings[index_b,:] += model([v_win_list_b, labels_b, labels_b_set])[-1].detach().cpu().squeeze().numpy()

    id_list = [PAR['word2id'][str(i)] for i in PAR['gl']]
    embeddings = embeddings[id_list]
    PAR['embeddings'] = embeddings
    return PAR