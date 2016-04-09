
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from pprint import pprint


# In[8]:

# Create Lookup Table Requires Q(|V| x numColumns) memory
# Format is Name,InNodes,TaskCost
table = [["Geneve",0,0], ["Hambourg",1,50], ["Londres",1,60], ["Edimburg",1,90], 
["Oslo",4,90], ["Amsterdam",3,80], ["Berlin",1,80], ["Helsinki",2,50], ["Stockholm",3,0]]
cities=[i[0]  for i in table]
innodes=[i[1] for i in table]
taskcost=[i[2] for i in table]
col_labels=["Cities","InNodes","TaskCost", "CriticalCost", "PreviousTask"]
df_lookup = pd.DataFrame(0,index=np.arange(len(table)),columns=col_labels)
df_lookup.InNodes  = innodes
df_lookup.Cities   = cities
df_lookup.TaskCost = taskcost 
df_lookup.index.name = "City"
df_lookup


# In[9]:

# Create Adjacent List 
# Required Memory: |V| x max(2*k)
# Format is Adjacent Name Node, Edge Cost
adj_neighbors = {}
adj_neighbors["Geneve"]    = [("Hambourg",680), ("Londres",360), ("Amsterdam",300)]
adj_neighbors["Hambourg"]  = [("Helsinki",100), ("Berlin",90)]
adj_neighbors["Londres"]   = [("Edimburg",60)]
adj_neighbors["Edimburg"]  = [("Amsterdam",220), ("Oslo",630), ("Stockholm",550)]
adj_neighbors["Oslo"]      = [("Stockholm",130)]
adj_neighbors["Amsterdam"] = [("Oslo",720)]
adj_neighbors["Berlin"]    = [("Helsinki",110), ("Oslo",210), ("Amsterdam",190)]
adj_neighbors["Helsinki"]  = [("Oslo",120), ("Stockholm",400)]
adj_neighbors["Stockholm"] = []
pprint(adj_neighbors)


# In[10]:

# search critical paths + topological sort
# required temporary additional memory: ~ Q{Start|V| + |V|}
# processing time: O(n*k) , k=max(3) per adjacent neighbors per vertex
def search_critical_paths(df, neighbors):
    # Find Starting Point of "InNodes" == 0)
    # Allocate Queue based on maximum number of vertices, as we should never resize the array
    start_vertices = list(df[df.InNodes == 0].Cities)
    queue = [0] * len(df)
    curr_queue_idx = 0;
    # Insert "InNodes" == 0 as Starting vertices
    for i,vertex in enumerate(start_vertices): queue[i] = vertex
    queue_count = len(start_vertices)
    
    while curr_queue_idx < queue_count:
        # retrieve certice from queue and lookup table
        vertice = queue[curr_queue_idx]
        v_idx = df[df.Cities == vertice].index
        df.ix[v_idx, "CriticalCost"] = df.ix[v_idx, "CriticalCost"] + df.ix[v_idx, "TaskCost"] 
        # iterate through neighbors of current vertex
        for neighbor,edge_cost in neighbors[vertice]: 
            # for each neighbor of current vertice, decrement index
            n_idx = df[df.Cities == neighbor].index        
            df.ix[n_idx, "InNodes"] -= 1
            # add in the edge cost of the neighbor
            p_critical = df.loc[v_idx, "CriticalCost"].values
            n_critical = df.loc[n_idx, "CriticalCost"].values
            if (p_critical + edge_cost) > n_critical: 
                df.ix[n_idx, "CriticalCost"] = p_critical + edge_cost
                df.ix[n_idx, "PreviousTask"] = vertice
        # find new "InNodes = 0" and mark item to be added to queue if not already in queue
        next_vertices = list(df[df.InNodes == 0].Cities)    
        for v in next_vertices: 
            if v not in queue:
                queue[queue_count] = v
                queue_count += 1
        # move to next element in queue
        curr_queue_idx += 1
    return queue

topological_sorted = search_critical_paths(df_lookup, adj_neighbors)   
print "Topological Sort Order Sequence:\n{0}".format(topological_sorted)
print "Critical Cost:\nCity:{0}, CriticalCost:{1}".format(df_lookup.ix[8,"Cities"], df_lookup.ix[8,"CriticalCost"])
df_lookup

