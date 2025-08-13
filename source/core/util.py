import numpy as np
import matplotlib.pyplot as plt
import teaspoon.MakeData.PointCloud as makePtCloud
from ipywidgets import interact, IntSlider
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
import teaspoon.TDA.Draw as Draw

def plot_vietoris_rips(epsilon, points_data):

    plt.figure(figsize=(8, 8))
    plt.scatter(points_data[:, 0], points_data[:, 1], c='blue', s=10)
    plt.title(f'Vietoris-Rips complex ε = {epsilon}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)
    plt.gca().set_aspect('equal', adjustable='box')

    dist_matrix = squareform(pdist(points_data))
    

    edges = np.argwhere(dist_matrix <= epsilon)
    
    for i, j in edges:
        if i < j:
            p1 = points_data[i]
            p2 = points_data[j]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', alpha=0.5)
    plt.axis('off')
    plt.show()

def DoubleAnnulus(N1=None,N2=None, r1 = 1, R1 = 2, r2 = .8, R2 = 1.3, xshift = 3):
    if N1 is not None:
        P = makePtCloud.Annulus(N=N1, r = r1, R = R1)
    else:
        P = makePtCloud.Annulus(r = r1, R = R1) 
    if N2 is not None:
        Q = makePtCloud.Annulus(N=N2, r = r2, R = R2)
    else:
        Q = makePtCloud.Annulus(r = r2, R = R2)

    Q[:,0] = Q[:,0] + xshift
    P = np.concatenate((P, Q) )
    return(P)


def drawPersistentDiagram(diagrams, R = 2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (20,5))
    # Draw diagrams
    plt.sca(axes[0])
    plt.title('0-dim Diagram')
    Draw.drawDgm(diagrams[0])

    plt.sca(axes[1])
    plt.title('1-dim Diagram')
    Draw.drawDgm(diagrams[1])
    plt.axis([0,R,0,R])
