This is the implementation for NodeSketch in MATLAB with C (see the following paper)

- Yang, Dingqi, Paolo Rosso, Bin Li, and Philippe Cudré-Mauroux. "NodeSketch: Highly-Efficient Graph Embeddings via Recursive Sketching." In KDD’19.

How to use (Tested on MATLAB 2017a and 2017b on MacOS and Ubuntu):
1. Compile sketch_node_embs_fast.c using mex (in MATLAB): 
mex CFLAGS='$CFLAGS -Ofast -march=native -ffast-math -Wall -funroll-loops -Wno-unused-result' sketch_node_embs_fast.c

2. Run experiment_NodeSketch.m

Please cite our paper if you publish material using this code.

