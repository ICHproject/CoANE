import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from sklearn.metrics import f1_score, accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from sklearn import preprocessing
from sklearn.metrics.cluster import v_measure_score
from sklearn.cluster import KMeans
#based on node2vec: https://github.com/lucashu1/link-prediction
def Edge_Embeddings(edges, embeddings):# word2id

  train_edges = edges["train_edges"]
  train_edges_false = edges["train_edges_false"]
  val_edges = edges["val_edges"]
  val_edges_false = edges["val_edges_false"]
  test_edges = edges["test_edges"]
  test_edges_false = edges["test_edges_false"]

  emb_list = []
  emb_matrix = embeddings


  def get_edge_embeddings(edge_list):
      embs = []
      for edge in edge_list:
          node1 = edge[0]
          node2 = edge[1]
          emb1 = emb_matrix[node1]
          emb2 = emb_matrix[node2]
          edge_emb = np.multiply(emb1, emb2)
          embs.append(edge_emb)
      embs = np.array(embs)
      return embs

  Edge_Embeddings_out = {}

  # Train-set edge embeddings
  pos_train_edge_embs = get_edge_embeddings(train_edges)
  neg_train_edge_embs = get_edge_embeddings(train_edges_false)
  Edge_Embeddings_out["train_edge_embs"] = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

  # Create train-set edge labels: 1 = real edge, 0 = false edge
  Edge_Embeddings_out["train_edge_labels"] = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

  # Val-set edge embeddings, labels
  pos_val_edge_embs = get_edge_embeddings(val_edges)
  neg_val_edge_embs = get_edge_embeddings(val_edges_false)
  Edge_Embeddings_out["val_edge_embs"] = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
  Edge_Embeddings_out["val_edge_labels"] = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

  # Test-set edge embeddings, labels
  pos_test_edge_embs = get_edge_embeddings(test_edges)
  neg_test_edge_embs = get_edge_embeddings(test_edges_false)
  Edge_Embeddings_out["test_edge_embs"] = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

  # Create val-set edge labels: 1 = real edge, 0 = false edge
  Edge_Embeddings_out["test_edge_labels"] = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

  return(Edge_Embeddings_out)

def evaluation(PAR, verbose = True):
    if len(PAR['edges'])==0:
        print("Can't find evaluation edge")
        return

    edges = PAR['edges']
    word2id = PAR['word2id']
    gl = PAR['gl']
    dict_NtoC = PAR['dict_NtoC']

    embeddings = PAR['embeddings']
    Edge_Embeddings_out = Edge_Embeddings(edges, embeddings)

    edge_classifier = LogisticRegression(random_state=0, solver= 'liblinear')
    edge_classifier.fit(Edge_Embeddings_out["train_edge_embs"], Edge_Embeddings_out["train_edge_labels"])

    train_preds = edge_classifier.predict_proba(Edge_Embeddings_out["train_edge_embs"])[:, 1]
    train_roc = roc_auc_score(Edge_Embeddings_out["train_edge_labels"], train_preds)

    #----------------------------------------------------------------------------

    # Predicted edge scores
    val_preds = edge_classifier.predict_proba(Edge_Embeddings_out["val_edge_embs"])[:, 1]
    val_roc = roc_auc_score(Edge_Embeddings_out["val_edge_labels"], val_preds)

    test_preds = edge_classifier.predict_proba(Edge_Embeddings_out["test_edge_embs"])[:, 1]
    test_roc = roc_auc_score(Edge_Embeddings_out["test_edge_labels"], test_preds)
    
    accuracy = [np.round(train_roc,4), np.round(val_roc,4), np.round(test_roc,4)]
    if verbose:
        print(np.round(accuracy, 3))
    return accuracy
