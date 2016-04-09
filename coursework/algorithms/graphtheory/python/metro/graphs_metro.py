
# coding: utf-8

# In[1]:

#%matplotlib inline
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pprint import pprint
import fibonacci_heap_mod as fib
import codecs


# In[2]:

def read_data_codec(filename):
    lines = ""
    with codecs.open(filename, "r", "ISO-8859-1" ) as f:
        lines = f.readlines()
    return lines

def read_data(filename):
    lines = ""
    with open(filename, "r") as f:
        lines = f.readlines()
    return lines
    
def parse_data(lines):
    nodes_d = {}
    edges_d = {}
    node_state = False
    edge_state = False

    for content in lines:
        content = content.strip().rstrip('\r\n')
        # alternatively go off the number of data points in a line (but this could easily be modified in the future)
        # in a distributed system, we might not have the state and need to go off the length or some other indication
        if content == "[Vertices]":   
            node_state = True    
            edge_state = False
            continue
        elif content == "[Edges]":
            edge_state = True
            node_state = False
            continue

        if(node_state):
            # split into identifier, name
            #print unicode()
            #content = content.encode('utf8')
            identity, name = content.split(' ', 1)
            nodes_d[int(identity)] = name
        elif(edge_state):
            # split into src,destination, weight
            src,dest,weight = content.split()
            """
            # equivalent to i=src < j=dest
            if (src < dest):
                curr_val = edges_d.get(int(dest), [])
                curr_val.insert(0, [src,weight])
                #curr_val.append([src,weight])
                edges_d[int(dest)] = curr_val
            # equivalnt to i=src > j=dest
            else:                
                curr_val = edges_d.get(int(src), [])
                curr_val.insert(0, [dest,weight])
                #curr_val.append([dest,weight])
                edges_d[int(src)] = curr_val
            """
            curr_val = edges_d.get(int(src), [])
            curr_val.append([dest,weight])
            edges_d[int(src)] = curr_val

    return nodes_d, edges_d

def generic_parse_data(lines):
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
            curr.append(content)
            data[active_section_name] = curr
    return data

def lookup_transform_vertice(vertice_name, nodes, duplicates):
    return nodes[nodes.name == vertice_name].id.values[0] 

def transform_to_dict(data):
    transformed_data = {}
    for k,v in data.items():
        for k2,v2 in v:
            curr = transformed_data.get(str(k), dict())
            curr[str(k2)] = v2
            transformed_data[str(k)] = curr
    return transformed_data

def create_duplicates_table(data):
    nodes_data = pd.DataFrame(nodes.items(), columns=["id","name"])
    duplicates = nodes_data[nodes_data["name"].duplicated()]
    duplicates_sta_names = set(duplicates.name)
    duplicates_table = nodes_data[nodes_data['name'].isin(duplicates.name)]
    return duplicates_table
    
def create_adj_dataframe(nodes, edges):
    cols = [str(i) for i in nodes.keys()]
    df = pd.DataFrame([], index=cols, columns=cols)
    df = df.fillna(float("inf"))
    # insert into row,col
    # iterate through edges and enable each one
    for k,v in edges.items():
        for k2,v2 in v:
            df.ix[k,k2] = float(v2)
        df.ix[k,k] = int(0)
    return df

def create_adj_matrix(nodes, edges):
    m_size = len(nodes.keys())
    m_adj = [[float("inf") for x in xrange(m_size)] for x in range(m_size)] 
    # iterate over each node in the edge to create adjacency matrix
    for k,v in edges.items():
        for k2,v2 in v:
            m_adj[k][int(k2)] = float(v2)
        m_adj[k][k] = int(0)
    return m_adj

def analyze_metrics(nodes, matrix):
    # Create a counter for all the neighboring vertices for each node (input=adj_df)
    cols = [str(i) for i in nodes.keys()]
    stats = pd.DataFrame([], index=cols, columns=['counter'])
    for c in matrix.columns:
        valid_col_weights = adj_df[(matrix[c] !=float(0)) & (matrix[c] != float("inf") )][c]
        num_valid_col_weights = len(valid_col_weights)
        stats.ix[int(c)]["counter"] = num_valid_col_weights
    stats = stats.sort(["counter"],ascending=False)
    
    # for each row find rows columns that are not '0'
    most_frequent = stats.counter
    most_frequent[:10].plot(kind='barh')
    plt.title("Most Frequent Visited Nodes/Stations")
    plt.xlabel("Number of Visits")
    plt.ylabel("Node/Station Num")
    
def save_to_np_format(matrix, filename='graphs_adj_matrix.txt'):
    # save data to file (input = adj_matrix)
    np.savetxt(filename, np.array(matrix), fmt='%1.2f')
    
def log_samples(vertices, edges, transformed_edges, count=5):
    print "Vertices: {id, name}:"
    pprint(vertices.items()[:count])
    print 
    print "Edges from Vertices: {src, [dest,cost]:}"
    pprint(edges.items()[:count])
    pprint(transformed_edges.items()[:count])


# In[3]:

# read and parse the data into dictionaries
# roughly calculate hhow long this takes with timeit
d_lines = read_data_codec("./metro_complet.txt")
nodes, edges = parse_data(d_lines)
edges_transf = transform_to_dict(edges)
df_duplicate_stas = create_duplicates_table(nodes)
df_nodes = pd.DataFrame(nodes.items(), columns=["id","name"])
#log_samples(nodes, edges, edges_transf)


# In[4]:

# create an adjacent matrix
adj_matrix=create_adj_matrix(nodes, edges)
query_row=7
query_cols = [int(k) for k,v in edges[query_row]]
for i in query_cols: print 'verify records: vertice={0},edge={1},cost={2}'.format(query_row,i,adj_matrix[query_row][i])


# In[5]:

# create an adjacent dataframe for easy manipulation
pd.set_option('display.max_columns', len(nodes.keys()))
adj_df=create_adj_dataframe(nodes, edges)
query_row=7
query_cols = [int(k) for k,v in edges[query_row]]
for i in query_cols: print 'verify records: vertice={0},edge={1},cost={2}'.format(query_row,i,adj_df.ix[query_row][i])


# In[6]:

# view the head adjacent_matrix dataframe
adj_df.head()


# In[7]:

# instead of multidimensional array, create a single vector to track
# TBD: Duplicates in Sparse Matrix
def bfs_traversal(G, v, duplicates):
    queue = []
    visited = []
    # Add the selected starting selection node to the queue
    queue.append(v)
 
    while len(queue) > 0:
        # get the first element in the queue
        item = int(queue.pop(0))
        # Handle the case where multiple nodes have same STA Name due to multiple 'track' lines
        # In this case merge all these nodes into a single adjacency neighbor list and process the node
        # This procedure would be done on a minimized graph based on the graph neighbor connected nodes
        if item in(duplicates.name):
            curr = duplicates[duplicates.id == item]
            sub_table = duplicates[duplicates['name'].isin(curr.name)]
            sta_id_duplicates = list(sub_table.id)
            curr_adjtable = G[item]
            unique_ids = set()
            # get all the unique adjacent neighbors into a single list from each of the duplicate stations
            for sta_id in sta_id_duplicates:
                for k,v in G[sta_id]:
                    unique_ids.add(int(k))                    
            unique_ids = unique_ids.difference(sta_id_duplicates)
            # mark current node, and duplicate nodes as visited (adjacent neighbors to be added to queue)
            for i in sta_id_duplicates: 
                if i not in visited: visited.append(i)
            for k in unique_ids:
                if k not in visited: queue.append(k)
        else:
            # mark the current process item as visited
            if item not in visited: visited.append(int(item))
            # add each of the adjacent neighbors to the queue
            for k,v in G[item]:
                if int(k) not in visited: queue.append(int(k))
    return visited

# determine if the graph is completely connected, based on all vertices visited
selection_vertex = 16
visited_nodes = bfs_traversal(edges, selection_vertex, df_duplicate_stas)
print "Num Vertices visited:{0}, Connected Graph:{1}".format(len(visited_nodes), len(visited_nodes) == len(nodes.keys()) ) 


# In[8]:

# priority queue to implement both a basic queue and fibonacci heap priority queue
class PriorityQueue(object):
    def __init__(self, queue_type="default"):
        self.queue_format = queue_type
        self.basic_queue  = {}
        self.fib_queue = fib.Fibonacci_heap()
    def initialize(self,src,src_val,key,val):
        set_val = val
        if key == src: set_val = src_val
        if self.queue_format == "fib":
            self.fib_queue.enqueue(key, set_val)
        else:
            self.basic_queue[key] = set_val
    def empty(self):
        if self.queue_format == "fib":
            if len(self.fib_queue): return False
            else: return True
        else:
            return not(len(self.basic_queue) > 0)
    def enqueue(self, key, priority):
        #print "key,priority",key,priority
        if self.queue_format == "fib": self.fib_queue.enqueue(key, priority)
        else: self.basic_queue[key] = priority
    def deque(self, dist):
        if self.queue_format == "fib":
            entry = self.fib_queue.dequeue_min()
            return entry.get_value() #,entry.get_priority
        else:
            u = min(self.basic_queue, key=lambda x: dist[x])  
            self.basic_queue.pop(u)
            return u  #,dist[u]


# In[9]:

def dijkstra(G, src, dest, queue_method="default"):
    dist = {}                         #dictionary of distances
    pred = {}                         #dictionary of previous visits
    Q = PriorityQueue(queue_method)   #priority queue implementation (default case does is just and unordered list)
    visited = set()                   #keep track of which nodes have been visited
    
    # initialization 
    for idx,v in enumerate(G):
        dist[v] = sys.maxint
        pred[v] = -1
        Q.initialize(src, 0, v, dist[v])        
    dist[src] = 0
        
    while not Q.empty():
        # get the minimum vertice, distance from the Q
        # the default implementation is using an unordered list, overrides can use a priority queue or fibonacci heap
        u = Q.deque(dist)
        visited.add(u) 
        # for each adjacent neighbor of u, w=distance from u to v
        for v,w in G[str(u)].items():
            if v in visited: continue
            new_dist = dist[u] + float(w)
            # determine if the new distance is shorter than last known path, if so update the new shorter path
            if( new_dist < dist[v] ):
                Q.enqueue(v, new_dist)
                dist[v]    = new_dist
                pred[v]    = u
    return dist, pred

def shortest_basic(pred, idx, path):
    while pred[idx] != -1:
        path.append(pred[idx])
        idx = pred[idx]
    return path

# TBD: Dataframe implementation not correctly implemented yet
def shortest(frame, idx, path):
    while str(idx) != str(-1):
        v = frame.ix[int(idx), "station"]
        path.append(v)
        idx = frame.ix[int(v), "predecessor"]
    return path


# In[10]:

cmd_args = read_data("./metro_input.txt")
sections_data = generic_parse_data(cmd_args)
src_sta_name = sections_data['[source station]'][0]
target_sta_name = sections_data['[destination station]'][0]
queue_type = sections_data['[Priority Queue Type]']
queue_type = [i for i in queue_type if i[0]!= '#'][0]
queue_type = "fib" if queue_type == "fibonacci" else "default"


# In[11]:

#  Test Case: perform dykstra shortest path on stations
#  Read from File for inputs
src_id  = lookup_transform_vertice(unicode(src_sta_name, "utf8"), df_nodes, df_duplicate_stas)
target_id = lookup_transform_vertice(target_sta_name, df_nodes, df_duplicate_stas)
print "\nRequested Source->{0}: {1}, Destination->{2}: {3}\n".format(src_id,src_sta_name,target_id,target_sta_name)
dist, pred = dijkstra(edges_transf, str(src_id), str(target_id), queue_type)

#%timeit dijkstra(edges_transf, str(src_id), str(target_id))
#%timeit dijkstra(edges_transf, str(src_id), str(target_id), "fib")

# basic results from shortest path
shortest_path = shortest_basic(pred, str(target_id), [str(target_id)])
shortest_path = shortest_path[::-1]
# transform convert results (ordered)
print "Shortest Path Route:"
for i in shortest_path: print i, df_nodes.ix[int(i),"name"]
# show the cost
print; print "Associated Cost:"
print target_id, dist[str(target_id)]


# In[12]:

# Merge the data ainto a single data frame
df_results = pd.DataFrame(pred.items(), columns=['station', 'predecessor'])
df_results = df_results.convert_objects(convert_numeric=True)
df_results = df_results.sort(columns='station', ascending=True)
df_results = df_results.reset_index()
df_results = df_results.drop(['index'],axis=1)
df_cost = pd.DataFrame(dist.items(), columns=['station', 'cost'])
df_cost = df_cost.convert_objects(convert_numeric=True)
df_results = df_results.merge(df_cost, on='station')
df_results["station name"] = df_results.station.apply(lambda x: df_nodes.ix[x, "name"])
df_results["predecessor name"] = df_results.predecessor[df_results.predecessor != -1].apply(lambda x: df_nodes.ix[x, "name"])
df_results = df_results.reindex_axis(["station", "station name", "predecessor", "predecessor name", "cost"],axis=1)
df_results.index.name = "station id"
df_results = df_results.ix[[int(i) for i in shortest_path]]
df_results["predecessor name"] = df_results["predecessor name"].astype('unicode')
df_results.head(25)


# In[13]:

# log results to output file
print "Writing output to file: metro_output.txt"
header = str(list(df_results.columns)) + "\r\n"
output = header
for k,v in df_results.iterrows():
    l = str(v.station) + "\t\t" + v["station name"] + ", "+ str(v.predecessor) + ", " +         v["predecessor name"] + ", " + str(v.cost) + "\r\n"
    output += l
with codecs.open("./metro_output.txt", 'w', "ISO-8859-1") as f: f.write(output)


# ######General Analysis (Dijkstra Algorithm) 
# * Memory Requirement per Priority Queue or standard queue: O(|V|)
# * Usage of an adajacent list, rather than an adjacent matrix.  Optimization of the adjacent list can further be performed, as each duplicates of neighbors are existent and need to be checked.
# * In regards to the adjacent list, a typical unsorted queue implementation, contributes runtime O(|V|^2).  If instead a priority queue is utilized, this can be improved upon to be O(|V| ln |V|).  Overall the total runtime contributes O(|E| ln |V|), due to additional visits of an adjacent edge.
# * Reviewing the analysis from '%timeit' shows that the runtime performed has been improved with a fibonacci priority queue, rather than manually sorting an unsorted array.

# In[ ]:



