import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
np.random.seed(0)

def Scan_Context_process(PAR, args):
    NUM_WALKS = args.num_walks
    verbose=args.verbose
    seed = args.seed
    window, nb_negative = args.window_hsize, args.num_negative
    train_sen, word2id, id2word, subsamples = PAR['cata_sentences'], PAR['word2id'], PAR['id2word'], PAR['subsamples']
    num_node = PAR['nb_id']#len(word2id.keys())+1
    mulfn = []

    mulDmatrix = []
    mulComatrix_1hop =[]

    mulreformfnlist = []
    mulreformfn = []

    mulcount_list = []
    mulnegFre = []
    mulnegFrea = []
    Max_win_count = 0 

  
    if verbose == True:
        print('Scaning Contexts...')
    #sentences in walks

    #initial
    fn = []
    reformfn = {x:[] for x in range(num_node)}
    window_count = {x:0 for x in range(num_node)}
    Comatrix = (np.zeros((num_node, num_node)))
    Comatrix_1hop = (np.zeros((num_node, num_node)))
    Win_stack = {}
    if len(train_sen)>1:
        train_sen = [train_sen]
    for sens in (train_sen):
        # if NUM_WALKS<l:
        #     break
        ##sentence in sentences
        for d in tqdm(sens):
            d = [0]*window + [word2id[w] for w in d if w in word2id] + [0]*window #padding

            np.random.seed(seed)
            r = np.random.random(len(d))#subsampling

            ####window in sentence
            for i in range(window, len(d)-window):
                #subsampling
                if d[i] in subsamples and r[i] > subsamples[d[i]] and i>window:
                    continue
                #End subsampling
                
                #save window in sequence form
                win = d[i-window:i+1+window]    

                #save window in node-wise form
                reformfn[d[i]].append(win)
                window_count[d[i]]+=1

                #co-cuurence matrix
                c = win[window]
                for w in win[:window]:
                    Comatrix[c, w] += 1
                Comatrix_1hop[c, w] += 1
                for w in win[(window+1)::]:
                    Comatrix[c, w] += 1
                Comatrix_1hop[c, win[window+1]] += 1
            ###End window in sentence
        ##End sentence in sentences
    #End sentences in walks
    #...............................
    mulfn.append(sp.csr_matrix(fn))
    fn = None
    # Comatrix_1hop[:, 0] *=0 # SK
    mulComatrix_1hop.append(sp.csr_matrix(Comatrix_1hop))
    Comatrix_1hop =None
    #...............................

    #check
    #print("Check...")
    #for k, v in reformfn.items():
    #    if len(v)==0:
    #        print(k)

    
    #update Max_win_count
    if verbose == True:
        print('Counting Contexts...')
    
    #update Max_win_count for NUM_WALK >1
    cur_max = max(window_count.values())
    Max_win_count = cur_max if Max_win_count< cur_max else Max_win_count


    #number of contexts per node
    count_list = [window_count[i] for i in range(num_node)]
    count_list_sort = sorted(count_list)[::-1]

    
    #Neg sampling
    TotalF = sum(np.array(count_list)) #total window_count
    negFre = [[0]* nb_negative] #neg samples padding for id 0
    negFrea = [] #controller
    
    #set context co-occurance prob.
    ALL_Win = [[j,(f)] for j,f in enumerate(count_list)]
    ALL_Win = np.array(ALL_Win)


    #Presampling Neg samples id
    np.random.seed(seed)
    Preindex = np.random.choice(ALL_Win[:, 0], nb_negative+10, replace=False, p=[(x/sum(ALL_Win[:, 1])) for x in ALL_Win[:, 1]])
    Preindex = [int(x) for x in Preindex]
    
    #Neg sampling controller
    indexa = np.array(count_list_sort[0])/TotalF
    #indexa = indexa.tolist()
    
    #choose neg samples out of contexts
    if verbose == True:
        print('Neg.Sampling...')
        
    for i in range(1, num_node):
        index = [j for j in Preindex if i!=j and Comatrix[i,j]==0][:nb_negative]
        ##if neg samples are not enough 
        while len(index) < nb_negative:
            x=int(np.random.choice(ALL_Win[:, 0], 1, replace=False, p=[(x/sum(ALL_Win[:, 1])) for x in ALL_Win[:, 1]]))
            if i!=x and Comatrix[i,x]==0:
                index.append(x)
        ##save id    
        negFre.append(sp.csr_matrix(index))
        index = None

    negFrea.append(indexa)

    #Normalized co-cuurence matrix
    # Comatrix[:, 0] *=0 # SK
    Dmatrix = Comatrix
    Comatrix = None
    Dmatrix[0] = (np.zeros(Dmatrix.shape[0]))
    Dmatrix[0,0]=1
    n_base = np.sqrt(np.sum(Dmatrix, 1))
    Dmatrix = ((Dmatrix.T/n_base).T)
    n_base = None
    
    #...............................
    mulDmatrix.append(sp.csr_matrix(Dmatrix))
    Dmatrix = None
    #...............................    
    mulcount_list.append(count_list)
    count_list = None
    mulnegFre.append(sp.vstack(negFre))
    negFre = None
    mulnegFrea.append(negFrea)
    negFrea = None 
    #...............................
    mulreformfnlist.append([reformfn[i] for i in range(len(reformfn))])
    reformfn = None
    #...............................

    if verbose == True:
        print('Summarizing...')

    mulreformfn = mulreformfnlist
    # for fnlist_id in range(len(mulreformfnlist)):
    #     fnlist = mulreformfnlist[fnlist_id]
    #     mulreformfnlist[fnlist_id] = None

    #     reformfn = (np.zeros((num_node, Max_win_count*(window*2+1))))
    #     ##each node
    #     for i in range(len(fnlist)):
    #       ##each window
    #       for j in range(len(fnlist[i])):
    #         reformfn[i, (window*2+1)*j:(window*2+1)*(j+1)] = fnlist[i][j]

    #     mulreformfn.append(sp.csr_matrix(reformfn))
    #     reformfn = None


    out_name = ["mulDmatrix", "mulreformfn", "mulcount_list", "Max_win_count", "mulnegFre", "mulnegFrea", "mulComatrix_1hop"]
    out_var =  [mulDmatrix, mulreformfn, mulcount_list, Max_win_count, mulnegFre, mulnegFrea, mulComatrix_1hop]
    out ={}
    for n, v in zip(out_name, out_var):
        out[n] = v
    if verbose == True:
        print('Contexts Generating Done!')
    return  out







