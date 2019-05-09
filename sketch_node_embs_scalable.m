function [embs_new] = sketch_node_embs_scalable(network, K_hash, Rand_beta, order_max, alpha)
% scalable implementation of nodesketch 
% network: SLA adjacancy matrix
% K_hash: embedding dimension
% Rand_beta: pre-generate hash value
% order_max: order of proximity
% alpha: decay weight


num_nodes = size(network,1);

if order_max<1
    error('proximity order should be an integer >=1');
end

if order_max>=1
    disp('###### NodeSketch 1st order...');
    tic;
    embs_old = sketch_node_embs_fast(network', K_hash, Rand_beta, 1, alpha);
    times(1) = toc;
end

embs_new = embs_old';

if order_max>=2
    [a,b,~] = find(network);
    embs_new = zeros(size(embs_old));
    tic;
    for order=2:order_max
        disp(strcat('##### NodeSketch order ',num2str(order),'...'));

        %     transforming sketches to network_sk and merge it with network
        
        a_new = repmat(a',K_hash,1);
        b_new = double(embs_old(:,b));
        network_sk = sparse(a_new(:),b_new(:),ones(numel(a_new),1),size(network,1),size(network,2));
        network_sk = network + network_sk*alpha/K_hash;
        embs_new = sketch_node_embs_fast(network_sk', K_hash, Rand_beta, 1, alpha);
%         sketching
        embs_old = embs_new;
        embs_new = embs_new';
    end
    
    
end

