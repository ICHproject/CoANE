import torch
import torch.nn as nn
import numpy as np
from CoANE.CoANE import *
from tqdm import tqdm
from CoANE.CoANE_Evaluation import *
torch.manual_seed(0)
np.random.seed(0)

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

def Train(PAR, Context, args):
    verbose = args.verbose
    seed = args.seed
    nb_node = PAR['nb_id']#Scan["mulreformfn"][0].shape[0]
    nb_filter = args.emb_dim #dimension of embedding

    win_len = 2*args.window_hsize+1
    max_win_count = Context["Max_win_count"]

    num_epochs = args.num_epochs
    V_BATCH_SIZE = min(args.num_Vbatch, nb_node)
    R_SIZE =  len(Context["mulreformfn"])
    device = args.device

    feat_dim = PAR['feat'].shape[1]
    
    #INPUT initial
    id_node = list(range(1, nb_node))
    num_mbatch = max(int(nb_node/V_BATCH_SIZE), 1)
    #Input initial
    X, Y = [], []

    #controller
    r2 = torch.from_numpy(np.array(Context["mulnegFrea"][0])).type(torch.FloatTensor).to(device)

    t_feat = torch.from_numpy(PAR['feat'].todense()).type(torch.FloatTensor).to(device)

    x_callfeat = torch.from_numpy(np.array(list(range(nb_node))).astype(int)).type(torch.LongTensor).to(device)
    for i in range(R_SIZE):
        #contexts
        # print(len(Context["mulreformfn"][i]), len(Context["mulreformfn"][i][0]))
        n_c = (args.window_hsize*2+1)
        # print(Context["mulreformfn"][i][:10])
        n_c_max = max([len(c) for c in Context["mulreformfn"][i]])
        
        Context_m = [sum(c + (n_c_max-len(c))*[[0]*n_c], []) for c in Context["mulreformfn"][i]]
        # assert 0
        x_reformfn = torch.LongTensor(Context_m).to(device)
        #torch.from_numpy(Context["mulreformfn"][i].todense().astype(int)).type(torch.LongTensor).to(device)
        Context["mulcount_list"][i][0]+=1
        #avg. par
        x_average_no = torch.from_numpy(np.array([1./(i if i else 1) for i in Context["mulcount_list"][i]])).type(torch.FloatTensor).to(device)
        #neg  samples
        x_negSample = torch.from_numpy(np.array(Context["mulnegFre"][i].todense()).astype(int)).type(torch.LongTensor).to(device) 
        #co-occurance matrix
        y_D = torch.from_numpy( Context["mulDmatrix"][i].todense()+Context["mulComatrix_1hop"][i].todense()).type(torch.FloatTensor).to(device)
        
        #
        x = [x_reformfn, x_average_no, x_negSample]
        y = [y_D]
        #
        X.append(x)
        Y.append(y)

    Context= None


    #MODEL initial
    model = CoANE(feat_dim, nb_filter, win_len, t_feat).to(device)
    mem_emb = nn.Embedding(nb_node, nb_filter).to(device)
    mem_emb.weight.requires_grad = False

    #Optimizer
    optimizer = torch.optim.Adam(list(model.parameters()), weight_decay = args.decay)
    print("weight_decay = ", args.decay)
    #Train model
    print('Training...')
    #if controller is not given
    if 'contoller' not in args:
        r2 = r2[0].detach().cpu().item()
    else:
        r2 = args.contoller
    
    for epoch in range(num_epochs):
        idxy = list(range(R_SIZE))
        np.random.shuffle(idxy)
        sentence_step = 0
        ##for each sentence
        for i in idxy:
            x = X[i]
            y = Y[i]

            ###Split V_Batch
            np.random.seed(seed)
            np.random.shuffle(id_node)

            mbatch, num_mbatch = Train_batch_processing(id_node, V_BATCH_SIZE)

            step = 0
            ###
            loss_r = []
            loss_r_p = []
            loss_r_n = []
            for j in mbatch:
                model.train()
                #Part 1--------------------------------------
                #Generate emb
                win_Encoder_feat,  feat_avg = model([x[0][j], x[1][j], t_feat])

                #Update memory emb
                mem_feat_avg = mem_emb(x_callfeat)
                mem_feat_avg[j] = feat_avg
                mem_emb = nn.Embedding.from_pretrained(mem_feat_avg)
                gather_neg_emb = mem_emb(x[2][j])

                #Part 2--------------------------------------
                #Loss
                loss_pos = (((torch.mm(feat_avg[:,:int(nb_filter/2)], torch.t(mem_feat_avg[1:,int(nb_filter/2):])).sigmoid()+1e-10).log()*-y[0][j,1:]).sum())
                loss_neg = (((torch.bmm(-feat_avg.view(-1, 1, nb_filter), torch.transpose(gather_neg_emb, 1,2)).squeeze()**2)*r2).sum())
                
                if args.c_f:
                    loss_mse = model.MSE(model.forward_f(feat_avg), t_feat[j])*args.c_f # original
                else:
                    loss_mse = loss_neg*0
                loss = loss_pos + loss_neg+loss_mse
                loss_r.append([loss.item(), loss_pos.item(), loss_neg.item(), loss_mse.item()])
                
                #Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step+=1
            ###           
            if verbose:
                loss_r = np.array(loss_r).sum(0)
                print(" Epoch [%d/%d], \n Avg. Loss:  {:%.2f} -p {:%.2f} -n {:%.2f} -m {:%.2f}" % ( epoch+1, \
                                            num_epochs, loss_r[0]/len(id_node),\
                loss_r[1]/len(id_node), loss_r[2]/len(id_node), loss_r[3]/len(id_node) ) )
                if epoch and (epoch+1)%5 ==0:
                    embeddings = np.zeros((nb_node, nb_filter))
                    m = V_BATCH_SIZE
                    with torch.no_grad():
                        for k in range(R_SIZE):
                            for l in tqdm(
                                        range(1,int(np.ceil(embeddings.shape[0]/m))+1), mininterval=2,
                                        desc='  - (renew-embedding)   ', leave=False):
                                embeddings[((l-1)*m+1):(l*m+1),:] += model([X[k][0][((l-1)*m+1):(l*m+1)], X[k][1][((l-1)*m+1):(l*m+1)], t_feat])[-1].detach().cpu().squeeze().numpy()
                        embeddings /= R_SIZE

                    id_list = [PAR['word2id'][str(i)] for i in PAR['gl']]
                    embeddings = embeddings[id_list]
                    PAR['embeddings'] = embeddings
                    accuracy = evaluation(PAR)

            sentence_step += 1 
            torch.cuda.empty_cache()
            
    #Renew
    print("Renewing embeddings...")
    embeddings = np.zeros((nb_node, nb_filter))
    m = V_BATCH_SIZE
    with torch.no_grad():
        for k in range(R_SIZE):
            for l in tqdm(
                        range(1,int(np.ceil(embeddings.shape[0]/m))+1), mininterval=2,
                        desc='  - (renew-embedding)   ', leave=False):
                embeddings[((l-1)*m+1):(l*m+1),:] += model([X[k][0][((l-1)*m+1):(l*m+1)], X[k][1][((l-1)*m+1):(l*m+1)], t_feat])[-1].detach().cpu().squeeze().numpy()
        embeddings /= R_SIZE

    id_list = [PAR['word2id'][str(i)] for i in PAR['gl']]
    embeddings = embeddings[id_list]
    PAR['embeddings'] = embeddings

    if verbose == True:
        print('Training Done!')
    return PAR, model