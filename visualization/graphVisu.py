import networkx as nx
import matplotlib.pyplot as plt

NEGATTENTION = 0.2


def graphVisual(attentionMatrix, agentNum, locVector):
    G = nx.DiGraph()
    pos = {}
    id = []
    # linewidth = []

    # creat all nodes
    for i in range(int(agentNum)):
        if i == 0 :
            G.add_node("E")
            pos["E"] = [0,0]
            id.append("E")
            nx.draw_networkx_nodes(G, pos, nodelist="E", node_size=1200, node_color='#ff0000')
        else:
            name = str(i)
            G.add_node(name)
            pos[str(i)] = locVector[i]
            id.append(name)
            nx.draw_networkx_nodes(G, pos, nodelist=str(i), node_size=700, node_color='#1f7814')
    

    # create all edges


    for i in range(int(agentNum)):
        for j in range(int(agentNum)):
            if attentionMatrix[i][j] > NEGATTENTION and i!=j:
                G.add_edge(id[i],id[j],weight=attentionMatrix[i][j])
                nx.draw_networkx_edges(G, pos, edgelist=[(id[i],id[j])], width=attentionMatrix[i][j]*8)
    
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    edge_labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels)
    ax = plt.gca()
    # ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    attention= [[0, 0.8, 0.5, 0.3, 0.25, 0.1],
                [0.7,0, 0.6, 0.1, 0.05, 0.03],
                [0.5,0.5,0, 0.2, 0.3, 0.05],
                [0.65,0.15,0.15,0,0.3,0.15],
                [0.55,0,0.4,0.3,0,0.3],
                [0.15,0,0,0.15,0.1,0]]
    location = [[0,0],[0,10],[3.5,5],[-3.5,-8],[3.5,-12],[0,-20]]
    num = 6
    graphVisual(attention,num,location)
