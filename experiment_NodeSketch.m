% NodeSketch
load('./graphs/blogcatalog.mat'); % best setting: order = 5; alpha = 0.001;
% load('./graphs/Homo_sapiens.mat'); % best setting: order = 2; alpha = 0.0001;
% load('./graphs/wiki.mat'); % best setting: order = 2; alpha = 0.0005;
% load('./graphs/dblp.mat'); % best setting: order = 4; alpha = 0.0002;
% load('./graphs/youtube.mat'); % best setting: order = 5; alpha = 0.002;


num_nodes = size(network,1);
K_hash = 128;
Rand_beta = -log(rand(K_hash,num_nodes)); % pre-generating hash value for efficiency

% order of proximity
order = 5;
% decay weight
alpha = 0.001;

%% NodeSketch normal implementation
% tic;
% embs = sketch_node_embs_fast(network'+speye(num_nodes), K_hash, Rand_beta, order, alpha);
% embs = embs'; % transpose due to the interface between MATLAB and C
% toc;

 
%% RECOMMENDED: NodeSketch scalable and fast implementation (merging process with MATLAB, even FASTER than using C!)
tic;
embs = sketch_node_embs_scalable(network+speye(num_nodes), K_hash, Rand_beta, order, alpha);
toc;

save('./embs_NodeSketch_blogcatalog.mat','embs');


%% Evaluation: Kernel SVM classification, modified from DeepWalk's evaluation code
% run the following code in the terminal
% python3 scoring_kernel_hamming.py --emb embs_NodeSketch_blogcatalog.mat --network graphs/blogcatalog.mat --outputfile res_NodeSketch_blogcatalog_classification_kernel.mat





