import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from util import load_data, separate_data
import matplotlib.colors

colors={0:'r',1:'orange',2:'yellow',3:'green',4:'lime',5:'blue',6:'magenta'}
#
#cmap = plt.cm.rainbow
#norm = matplotlib.colors.Normalize(vmin=0, vmax=5)

g_att=[]
i=0
np.random.seed(0)
graphs, num_classes = load_data("NCI1", False)
train_graphs, test_graphs = separate_data(graphs, 0, 0)
Gs = train_graphs[0:64]
load_path=input("input which file that you want read:")
att5=np.load(load_path)
for g in Gs:
    n = g.g.number_of_nodes()
    g_att.append(list(att5[0,i:i+n]))
    i=i+n
    
i=0
while(1):
    se = input("inout your select:")
    print(se)
    if se=="w":
        i=min(len(Gs)-1,i+1)
    else:
        if se == "s":
            i = max(0,i-1)
        else:
            break
    print("i")
    g=Gs[i]
    print("node tag:",g.node_tags)
    print("node attention:",g_att[i])
    pos= nx.spring_layout(g.g)
    labels={}
    k=0
    for m in g_att[i]:
        labels[k] = str(int(m*10000)/100)+"/"+str(k)
        k += 1
    
    plt.figure()
    temp=g.node_tags
    clo = [colors[min(m,6)] for m in temp]
    nx.draw_networkx_nodes(g.g, pos,
                       nodelist=list(range(g.g.number_of_nodes())),
                       node_color=clo,
                       node_size=400,
                       alpha=0.8)
    nx.draw_networkx_edges(g.g, pos, width=1.0, alpha=0.5) 
    nx.draw_networkx_labels(g.g, pos, labels, font_size=16)
    plt.show()
    
        
    
        