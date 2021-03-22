import argparse
import pickle
import torch

from CoANE.Data_Processing import *
from CoANE.Random_Walk import *
from CoANE.Contexts_Generator import *
from CoANE.CoANE_Train import *
from CoANE.CoANE_Evaluation import *

from time import time
import scipy.io
import pickle
import warnings
import datetime

warnings.filterwarnings('ignore')
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')


def parse_args():
    #Parses the arguments.
    parser = argparse.ArgumentParser(description="CoANE")
    parser.add_argument('--data_dir', type=str, default='./datasets', help='Directory of Datasets folder')
    parser.add_argument('--dataset', type=str, default='cora', help='Name of Datasets folder')
    parser.add_argument('--device', type=str, default='cuda', help='Training run on gpu("cuda") or cpu("cpu")')

    parser.add_argument('--verbose', type=bool, default=True, help='Print log')
    parser.add_argument('--reload', type=bool, default=False, help='Learning by assigned "PAR"&"Context" (exist in the data folder)')
    parser.add_argument('--save', type=bool, default=False, help='Save data & parameters ("network_tuple", "PAR", "HPAR", "Context" and "model")')
    parser.add_argument('--Spliting', type=bool, default=True, help='If data is needed to split')
    parser.add_argument('--dircted', type=bool, default=False, help='If network is directed')
    parser.add_argument('--eval', type=bool, default=True, help='Run evaluation')
    parser.add_argument('--eval_iter', type=bool, default=False, help='print evaluation for each epoch')

    parser.add_argument('--window_hsize', type=int, default=5, help='Size of half of window(context)')
    parser.add_argument('--emb_dim', type=int, default=128, help='Dimension of embedding')
    parser.add_argument('--num_epochs', type=int, default=15, help='Training epoch')
    parser.add_argument('--num_Vbatch', type=int, default=64, help='Number of nodes for each training epoch')
    parser.add_argument('--num_negative', type=int, default=20, help='Number of negative sampling nodes')
    parser.add_argument('--seed', type=int, default=0, help='Seed for any random function')
    parser.add_argument('--num_walks', type=int, default=1, help='Number of repeating sentence')
    parser.add_argument('--p', type=int, default=1, help='RandomWalk: biased parameter of node2vec (wide)')
    parser.add_argument('--q', type=int, default=1, help='RandomWalk: biased parameter of node2vec (deep)')
    parser.add_argument('--walk_length', type=int, default=80, help='RandomWalk: Length of sentence')
    parser.add_argument('--index', type=int, default=1, help='RandomWalk: Length of sentence')
    parser.add_argument('--star', type=int, default=1, help='RandomWalk: Length of sentence')
    parser.add_argument('--end', type=int, default=10, help='RandomWalk: Length of sentence')

    parser.add_argument('--contoller', type=float, default=0.001, help='Parameter of contoller of negative sampling')
    parser.add_argument('--c_f', type=float, default=1e5, help='Hyperparameter of attribute preservation')
    parser.add_argument('--decay', type=float, default=5e-2, help='Learning decay')
    parser.add_argument('--val_frac', type=float, default=.1, help='Split val data ratio')
    parser.add_argument('--test_frac', type=float, default=.2, help='Split test data ratio')
    parser.add_argument('--subsample_rate', type=float, default=1e-5, help='Subsample rate')

    return parser.parse_args() #add args=[] as argument when running on notebook

if __name__ == "__main__":
    args = parse_args()
    args.data = args.dataset
    #Data Processing
    try:
        network_tuple = load_obj(args.data_dir+"/{}/adj-feat-label".format(args.dataset))
    except:
        network_tuple = read_dataset(args)
    #Random Walk
    try:
        PAR = load_obj(args.data_dir+"/{0}/PAR/PAR_1".format(args.dataset))
        print("loading PAR: ", args.data_dir+"/{0}/PAR/PAR_1".format(args.dataset))
    except:
        PAR = random_walk(network_tuple, args)
  #Generating Contexts
    try:
        Context = load_obj("./Context/Context_{0}_{1}".format(args.dataset, args.window_hsize))
        print('Load.....Context ', "Context_{0}_{1}".format(args.dataset, args.window_hsize))
    except:
        Context = Scan_Context_process(PAR, args)
        print('Save.....')

    #Training
    T1 = time()
    PAR_emb, CoANE = Train(PAR, Context, args)
    T2 = time()
    print("Time Cost: ", T2-T1)

    #Evaluating Link Prediction
    accuracy = evaluation(PAR_emb)

    # save_obj(PAR_emb, "CoANE_{0}_emb".format(args.dataset))