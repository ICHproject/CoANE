# CoANE: Modeling Context Co-occurrence forAttributed Network Embedding

Code for paper "CoANE: Modeling Context Co-occurrence forAttributed Network Embedding". We provide the pytorch implement of the CoANE algorithm including Context Generation, Embedding learning for contexts and proximities preservation of two likelihoods for the attributed network embedding task.

![CoANE](https://github.com/ICHRick/CoANE/blob/master/CoANE_framework.JPG)


## Requirements
```
networkx==2.2
tqdm==4.50.2
numpy==1.19.1
torch==1.6.0+cu101
scipy==1.5.0
scikit_learn==0.23.2
```

## Basic Usage

### Input Data 
Each dataset contains 2 files which follow the format of Cora dataset: CITES and CONTENT.
```
1. cora.CITES: each line contains a edge relationship consisting of a head node, a space or tab and a tail node.
node_t_id node_h_id
node_t_id node_h_id
...

2. cora.feature: this file contain three parts, node_id, attributes and label.
Each line recorded by the following format:
node_id_1 feature_1 feature_2 ... feature_n label
node_id_2 feature_1 feature_2 ... feature_n label
...

```

### Output
If the argument '--save' is True, the embbedding matrix would be saved in the dictionary variable named "PAR_emb" with key 'embeddings':

embbedding matrix: node_number by embedding_dimension (in ascending order of node id)
```
Each line is as follows (id term is not included):
(node_id_1) dim_1, dim_2, ... dim_d'
(node_id_2) dim_1, dim_2, ... dim_d'
```
### Run
To  quickly run CoANE with default setting, execute the following command for link prediction evaluation for Cora Data:
```
python main.py
```

Check hyperparameters of CoANE:
```
  --data_dir        Directory of Datasets folder
  --dataset         Name of Datasets folder
  --device          Training run on gpu("cuda") or cpu("cpu")
  
  --verbose         Print log
  --Split        If data is needed to split
  --dircted         If network is directed
  
  --window_hsize    Size of half of window(context)
  --emb_dim         Dimension of embedding
  --num_epochs      Training epoch
  --num_Vbatch      Number of nodes for each training epoch
  --num_negative    Number of negative sampling nodes
  --seed            Seed for any random function
  --num_walks       Number of repeating sentence
  --p               RandomWalk: biased parameter of node2vec (wide)
  --q               RandomWalk: biased parameter of node2vec (deep)
  --walk_length     RandomWalk: Length of sentence
  
  --contoller       Parameter of contoller of negative sampling
  --val_frac        Split val data ratio
  --test_frac       Split test data ratio
  --subsample_rate  Subsample rate
```
Or run the following code:
```
python main.py -h
```
