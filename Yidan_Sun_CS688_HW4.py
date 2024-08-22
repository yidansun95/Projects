import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv("E:\\BU\\2024summer\\CS688\\week4\\person_knows_person.csv", sep='|')
data.head()

#Smaller dataset
data = data[0:1000]

#Empty Graph
graph = nx.Graph()

# Add edges from the dataset
for index, row in data.iterrows():
    graph.add_edge(row['Person.id'], row['Person.id.1'])

# Optionally, visualize the graph (requires matplotlib)
nx.draw(graph, with_labels=True, node_size=200, font_size=5, font_color='white',edge_color='grey',node_color='darkgreen')
plt.show()



# Dijkstra's algorithm-Shortest Path
source_node = 38
target_node = 570

if nx.has_path(graph, source_node, target_node):
    shortest_path = nx.dijkstra_path(graph, source=source_node, target=target_node)
    print(f"The shortest path from node {source_node} to node {target_node} is: {shortest_path}")
else:
    print(f"No path found between node {source_node} and node {target_node}.")



# A* Shortest Path
source_node = 38  # Replace with your actual source node
target_node = 570  # Replace with your actual target node

# Ensure there is a path between the source and target nodes
if nx.has_path(graph, source_node, target_node):
    # Use A* algorithm to find the shortest path
    shortest_path = nx.astar_path(graph, source=source_node, target=target_node)
    print(f"The shortest path from node {source_node} to node {target_node} using A* is: {shortest_path}")
else:
    print(f"No path found between node {source_node} and node {target_node}.")


# Compute the MST using Prim's algorithm
mst_prim = nx.minimum_spanning_tree(graph, algorithm='prim')

# Compute the MST using Kruskal's algorithm
mst_kruskal = nx.minimum_spanning_tree(graph, algorithm='kruskal')


# Prim's Removal
graph_trimmed_prim = graph.copy()

# Get the edges that are part of the MST (Prim's)
mst_edges_prim = set(mst_prim.edges())

# Remove edges that are not in the MST (Prim's)
for edge in list(graph_trimmed_prim.edges()):
    if edge not in mst_edges_prim and (edge[1], edge[0]) not in mst_edges_prim:
        graph_trimmed_prim.remove_edge(*edge)

# Visualize the trimmed graph for Prim's MST
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(graph_trimmed_prim)
nx.draw(graph_trimmed_prim, pos, with_labels=True, node_size=100, font_size=6, font_color='black',node_color='lightblue', edge_color='grey', width=2)
plt.title("Graph with only MST edges (Prim's)")
plt.show()


#Kruskal's Removal
graph_trimmed_kruskal = graph.copy()

# Get the edges that are part of the MST (Kruskal's)
mst_edges_kruskal = set(mst_kruskal.edges())

# Remove edges that are not in the MST (Kruskal's)
for edge in list(graph_trimmed_kruskal.edges()):
    if edge not in mst_edges_kruskal and (edge[1], edge[0]) not in mst_edges_kruskal:
        graph_trimmed_kruskal.remove_edge(*edge)

# Visualize the trimmed graph for Kruskal's MST
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(graph_trimmed_kruskal)
nx.draw(graph_trimmed_kruskal, pos, with_labels=True, node_size=100, font_size=6,font_color='white', node_color='darkblue', edge_color='grey', width=2)
plt.title("Graph with only MST edges (Kruskal's)")
plt.show()



# Compute PageRank
pagerank_scores = nx.pagerank(graph)

# Compute HITS (Hubs and Authorities)
hits_hubs, hits_authorities = nx.hits(graph)



# Extract values for normalization
pagerank_values = np.array(list(pagerank_scores.values()))
hits_hubs_values = np.array(list(hits_hubs.values()))
hits_authorities_values = np.array(list(hits_authorities.values()))

# Normalize the values
pagerank_normalized = (pagerank_values - pagerank_values.min()) / (pagerank_values.max() - pagerank_values.min())
hits_hubs_normalized = (hits_hubs_values - hits_hubs_values.min()) / (hits_hubs_values.max() - hits_hubs_values.min())
hits_authorities_normalized = (hits_authorities_values - hits_authorities_values.min()) / (hits_authorities_values.max() - hits_authorities_values.min())

def visualize_graph(graph, pos, node_colors, title, font_color='beige', node_cmap=plt.cm.viridis):
    plt.figure(figsize=(10, 10))
    # Use the specified colormap and font color
    nx.draw(graph, pos, with_labels=True, node_size=400, font_size=6,
            node_color=node_colors, cmap=node_cmap, edge_color='grey', font_color=font_color)
    plt.title(title)
    plt.show()


# Position for the nodes
pos = nx.spring_layout(graph)

# Visualize PageRank
node_colors_pagerank = [pagerank_scores[node] for node in graph.nodes()]
print("Visualizing PageRank Scores")
visualize_graph(graph, pos, node_colors_pagerank, "PageRank Scores Visualization")


# Visualize HITS Hubs
node_colors_hubs = [hits_hubs[node] for node in graph.nodes()]
print("Visualizing HITS Hubs Scores")
visualize_graph(graph, pos, node_colors_hubs, "HITS Hubs Scores Visualization")

# Visualize HITS Authorities
node_colors_authorities = [hits_authorities[node] for node in graph.nodes()]
print("Visualizing HITS Authorities Scores")
visualize_graph(graph, pos, node_colors_authorities, "HITS Authorities Scores Visualization")