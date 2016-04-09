
# coding: utf-8

# In[1]:

import pandas as pd
import numpy  as np
from pprint import pprint


# In[2]:

def read_data(filename):
    lines = ""
    with open(filename) as f:
        lines = f.readlines()
    return lines
# read in data file
d_lines = read_data("./drinking_network.inp")


# In[3]:

def parse_data(lines):
    name = ""
    data = {}
    active_section_state = False
    active_section_name  = ""
    for content in lines:
        content = content.strip().rstrip('\r\n')
        if len(content) == 0: continue
        # if this is content we filtered on, store away the key for processing and set the state
        if content[0] == '[': 
            curr = data.get(content, [])
            data[content] = curr
            active_section_state = True
            active_section_name  = content            
        elif active_section_state and content[0] != ';':
            curr = data.get(active_section_name, [])
            line = content.split()
            curr.append(line)
            data[active_section_name] = curr
    return data
            
sections_data = parse_data(d_lines)


# In[4]:

# filter the selected data based on the rules
def filter_data(data, sections):
    filtered = {}
    for k,v, in data.items():
        if k in sections:
            if k == "[JUNCTIONS]" or k == "[TANKS]" or k == "[RESERVOIRS]":
                for sample in v:
                    if sample[0][:2] != "NV": 
                        curr = filtered.get(k, [])
                        curr.append({"ID":sample[0]})
                        filtered[k] = curr
            elif k == "[PIPES]" or k == "[VALVES]":
                for sample in v:
                    curr = filtered.get(k, [])
                    curr.append({"ID": sample[0], "Node1":sample[1], "Node2":sample[2] })
                    filtered[k] = curr                
    return filtered
    
d_sections = ['[JUNCTIONS]','[PIPES]','[VALVES]', '[TANKS]', '[RESERVOIRS]'] 
subset_data = filter_data(sections_data, d_sections)
df_valves = pd.DataFrame(subset_data["[VALVES]"])
df_pipes  = pd.DataFrame(subset_data["[PIPES]"])
df_junctions = pd.DataFrame(subset_data["[JUNCTIONS]"] )
df_tanks = pd.DataFrame(subset_data["[TANKS]"])
df_reservoirs = pd.DataFrame(subset_data["[RESERVOIRS]"])
df_junctions = df_junctions.append(df_tanks)
df_junctions = df_junctions.append(df_reservoirs)


# In[5]:

def bfs_traversal(G, v, v_root, explore):
    queue = []
    visited = []
    # Add the selected starting selection node to the queue
    queue.append(v)
    while len(queue) > 0:
        # get the first element in the queue
        item = queue.pop(0)
        # mark the current process item as visited
        if item not in visited: visited.append(item)
        # add each of the adjacent neighbors to the queue
        for k,v in G[item].items():
            #print "searching:",k,v,v.label_id, v.explored
            if k == v_root: continue
            if k not in visited and not v.explored: queue.append(k)
            if v.valve_existent and not v.explored: 
                v.explored = 1
                G[k][item].explored =1                    
                explore.append(v.label_id)
                #explore.append((v.label_id, k))
                #print 'valve founder:', item, k,v, v.label_id, v.explored, explore                
                if v.explored: return visited, explore
    return visited,explore


# In[6]:

class AdjacentNeighbor(object):
    def __init__(self, v_id, label_id, pattern, state):
        self.pipe_existent  = False
        self.valve_existent = False
        self.vertex_id      = v_id
        self.label_id       = label_id
        self.root           = None
        self.explored       = 0
        self.assign(pattern, state)
        
    def assign(self, pattern, state):
        if pattern == "pipe": self.pipe_existent = state
        elif pattern == "valve": self.valve_existent = state

def assign_to_dict(frame, key, data):
    curr = frame.get(key, [])
    curr.append(data)
    frame[key] = curr
    return frame 

def search_adj_neighbor(frame, key, vertex_id, label_id, pattern, state):
    # get the current map for the vertex id
    curr = frame.get(key, dict())        
    adj  = curr.get(vertex_id, AdjacentNeighbor(vertex_id, label_id, pattern, state))
    adj.assign(pattern, state)
    curr[vertex_id] = adj
    frame[key] = curr
    return frame
    
def create_adjacent_neighbors(junctions, pipes, valves):
    # iterate through junction list, and find neighbors based on pipes and valves
    neighbors = {}
    for index,row in junctions.iterrows(): 
        pipe_node_neighbor  = pipes.loc[ (pipes.Node1 == row.ID) | (pipes.Node2 == row.ID)]
        valve_node_neighbor = valves.loc[ (valves.Node1 == row.ID) | (valves.Node2 == row.ID)]
        for index,node_i in pipe_node_neighbor.iterrows():
            nx =  filter(lambda x: x != row.ID, [node_i.Node1,node_i.Node2])[0]
            neighbors = search_adj_neighbor(neighbors, row.ID, nx, node_i.ID, "pipe", True)            
        for index,node_i in valve_node_neighbor.iterrows():
            nx =  filter(lambda x: x != row.ID, [node_i.Node1,node_i.Node2])[0]
            neighbors = search_adj_neighbor(neighbors, row.ID, nx, node_i.ID, "valve", True)
    return neighbors


# In[7]:

# create a sparse neighbor list
# iterate and find all vertices that matched together (Pipe: Set(Valves))
def create_pipe_valve_association(neighbors, pipes, valves, pipe_id):
    pipeline  = {}
    df_pipes_filtered = pipes[pipes["ID"].isin(pipe_id)]
    for index,row in df_pipes_filtered.iterrows():
    #for index,row in pipes.iterrows():
        # For a particular pipeline and corresponding nodes, find all existing valves
        # Default Case: If we find valves on each side of the junction - we are done
        valve_node1 = valves.loc[ (valves.Node1 == row.Node1) | (valves.Node2 == row.Node1)].ID.values
        valve_node2 = valves.loc[ (valves.Node1 == row.Node2) | (valves.Node2 == row.Node2)].ID.values
        valve_pipeline = list(valve_node1) + list(valve_node2)
        pipeline[row.ID] = set(valve_pipeline)

        # when haven't found a valve on each side of a junction point for the respective pipeline, walk the valves
        # handles the case where we had pipes on each side of the junction, and no direct valves
        # handles the case where we only need to one particular junction from this pipeline
        # there may exist a termination node where we don't need to handle a valve (will be noted in neighbor list)
        if len(valve_pipeline) < 2:
            # we drop the search from the valves we have already found from our search
            connpipe_node1 = pipes.loc[ (pipes.Node1 == row.Node1) | (pipes.Node2 == row.Node1) ] 
            connpipe_node1_sub = connpipe_node1.drop(index)
            connpipe_node2 = pipes.loc[ (pipes.Node1 == row.Node2) | (pipes.Node2 == row.Node2) ]
            connpipe_node2_sub = connpipe_node2.drop(index)
            # walk each of these pipes from the vertice in adjacency table 
            if len(connpipe_node1_sub):
                explored = []
                for k,v in adj_neighbors[row.Node1].items(): 
                    if v.pipe_existent: _, explored = bfs_traversal(neighbors, v.vertex_id, row.Node1, explored )
                for i in explored: pipeline[row.ID].add(i)
            if len(connpipe_node2_sub):
                explored = []
                for k,v in adj_neighbors[row.Node2].items(): 
                    if v.pipe_existent: _, explored = bfs_traversal(neighbors, v.vertex_id, row.Node2, explored )
                for i in explored: pipeline[row.ID].add(i)
    return pipeline



# In[8]:

# Parse input file and create valve association
# pipe_input = list( df_pipes.ID[:count].values )
cmd_args = read_data("./drinking_network_input.txt")
sections_data = parse_data(cmd_args)
pipe_selections = sections_data['[Pipe Queries]']
pipe_selections = [i[0] for i in pipe_selections if i[0]!= '#']
for i in pipe_selections: print "Pipe Queries: {0}".format(i)


# In[9]:

# Execute pipe association
adj_neighbors = create_adjacent_neighbors(df_junctions, df_pipes, df_valves)
pipeline_assoc = create_pipe_valve_association(adj_neighbors, df_pipes, df_valves, pipe_selections)
for k,v in adj_neighbors.items(): del adj_neighbors[k]
del adj_neighbors


# In[10]:

# output based on selection of pipe_filter
pd.set_option('display.max_colwidth', 1000)
df_pipeline = pd.DataFrame(pipeline_assoc.items(), columns = ['pipe', 'valves'])
df_pipeline.valves = df_pipeline.valves.apply(lambda x: ", ".join(x) )
df_pipeline.index.name = "query"
df_pipeline.head(5)


# In[11]:

# write output to file
# log results to output file
print "Writing output to file: drinking_network_output.txt"
header = list(df_pipeline.columns)
lines = [str(df_pipeline.ix[i,'pipe']) + "\t\t" + str(df_pipeline.ix[i,'valves']) for i,v in df_pipeline.iterrows()]
lines.insert(0, header)
output = ""
for l in lines: output += str(l) + "\r\n"
with open("./drinking_network_output.txt", 'w') as f: f.write(output)


# ######General Analysis
# - Pipeline Association : O(|V| + |E|), where E is for each adjacent neighbor of a given vertex v
# - BFS from pipeline association contributes O(|V| + |E|)
# - Each vertex for the Adacent list contributes memory, the size of AdjacentNeighbor class
# - Optimization in the Adjacency and usage of it can be made such that duplicates are not stored, and the corresponding neighbor entry is not duplicated, and only stored in one place.

# In[ ]:



