# degree_centrality.py
import networkx as nx
import pickle

def calculate_and_save_degree_centrality():
    # Load the saved monthly graphs
    with open('monthly_graphs.pkl', 'rb') as f:
        monthly_graphs = pickle.load(f)

    print("Calculating degree centrality...")
    degree_centralities = {month: nx.degree_centrality(G) for month, G in monthly_graphs.items()}

    # Save the degree centrality results
    with open('degree_centralities.pkl', 'wb') as f:
        pickle.dump(degree_centralities, f)
    print("Degree centrality saved.")
    return degree_centralities

calculate_and_save_degree_centrality()
