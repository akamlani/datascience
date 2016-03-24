fileID = fopen('edges.txt','r');
tline = fgets(fileID);

debug_ctr = 0;
adj_table = containers.Map;
while ischar(tline)
    %disp(tline);
    [C, matches] = strsplit(deblank(tline), ' ');
    snode = char(C{1});
    dnode = char(C{2});
    weight = str2double(C{3});
    % if the starting station node does NOT(~) exist insert a whole new map
    if ~(isKey(adj_table, {snode}))       
        meta = containers.Map;
        % access cell type data via {index} instead of (index)
        % get the destination node and weight as the meta value
        % use dictionary usage via containers.Map instead of struct()
        meta(dnode) = weight; 
        % insert adj_table with the starting node as the key
        adj_table(snode) = meta;
    % else appending to an already existing dictionary for the snode
    else
        % get the existing metadata of the snode and add the new data to it
        current_meta = adj_table(snode);
        current_meta(dnode) = weight;
    end
    tline = fgetl(fileID);
end
fclose(fileID);


% DEBUG: look at contents of data (multiline commented out: %{, %}
%{
disp(adj_table.keys);
k_srcnodes = adj_table.keys;
numkeys = size(k);
for ki_snode = k_srcnodes
    % for the given source node get all the neighbor keys
    disp(sprintf('source node key index=%s', char(ki_snode)));
    ki_snode_meta = adj_table(char(ki_snode));
    k_dnodes = adj_table(char(ki_snode)).keys;
    % iterate through each of the destination nodes from the src node
    for ki_dnode = k_dnodes
        disp(sprintf('dest node key index=%s', char(ki_dnode)));
        disp(ki_snode_meta(char(ki_dnode)));  
    end
end
%}
