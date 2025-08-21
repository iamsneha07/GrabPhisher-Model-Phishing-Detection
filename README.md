# 🕵️ GrabPhisher Model for Phishing Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Active-success)  

This project implements **GrabPhisher**, a phishing detection pipeline using **graph-based learning** techniques. It builds a **transaction graph** from raw data, generates **node embeddings** using **Graph Neural Networks** (GNN), and applies machine learning for phishing classification. The model leverages **Node2Vec** for representation learning and evaluates performance with rich visual metrics.

---

## ✅ Key Features  
✔ **Builds Transaction Graph** from CSV data  
✔ **Node Embeddings with Node2Vec** for feature representation  
✔ **Graph Construction & Analysis** using NetworkX  
✔ **Phishing Detection** using XGBoost Classifier  
✔ **Performance Metrics & Visualization** (Accuracy, Precision, Recall, F1-Score)  
✔ **Comparison Table** for baseline vs proposed model  
✔ **Progress Tracking with TQDM**  

---

## ⚙️ Tech Stack  
- **Python 3.8+**  
- [NetworkX](https://networkx.org/) – Graph processing  
- [Node2Vec](https://snap.stanford.edu/node2vec/) – Graph embeddings  
- [XGBoost](https://xgboost.readthedocs.io/) – Phishing classification  
- [Scikit-learn](https://scikit-learn.org/) – Evaluation metrics  
- [TQDM](https://github.com/tqdm/tqdm) – Progress visualization  

---

## 🛡️ Purpose  
GrabPhisher aims to **improve phishing detection accuracy** by modeling transaction data as a graph and capturing relational dependencies between nodes. Unlike traditional models that rely only on tabular features, this graph-based approach provides **context-aware detection** for robust security.  

---

## ✅ Project Status  
✅ **Working Prototype**  
✅ **Tested on Sample Transaction Data**  
✅ **Supports Integration with Larger Datasets**  

---

## 🔍 Future Enhancements  
✔ **Add Graph Neural Network (GNN) Encoder**  
✔ **Integrate Contrastive Learning for Better Embeddings**  
✔ **Enable Streaming Data Detection for Real-Time Phishing Alerts**  

---

## 🏆 Authors  
👤 **Sneha Deb**  
📧 [debsneha357@gmail.com](mailto:debsneha357@gmail.com)  

👤 **Shubhashree Bhar**  
📧 [shubhashreebhar@gmail.com](mailto:shubhashreebhar@gmail.com)  

---

### ⭐ Support  
If you like this project, give it a **star** ⭐ on GitHub!  
