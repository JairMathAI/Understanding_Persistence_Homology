import numpy as np
import matplotlib.pyplot as plt
import teaspoon.MakeData.PointCloud as makePtCloud
from ipywidgets import interact, IntSlider
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
import teaspoon.TDA.Draw as Draw
import gudhi
from persim import PersLandscapeApprox, PersLandscapeExact
from persim.landscapes import plot_landscape_simple
import tadasets

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


def plot_landscapes(Diagrams, n_func=11 ,exact=False):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 

    
    if not exact:
        pla0 = PersLandscapeApprox(dgms=Diagrams, hom_deg=0)
        plot_landscape_simple(pla0, ax=axs[0], depth_range=range(n_func))
    else:
        pla0 = PersLandscapeExact(dgms=Diagrams,hom_deg=0)
        plot_landscape_simple(pla0, ax=axs[0])



    axs[0].set_title('Persistence Landscape, H0 (Dimension 0)')

    

    if not exact:
        pla1 = PersLandscapeApprox(dgms=Diagrams, hom_deg=1)
        plot_landscape_simple(pla1, ax=axs[1], depth_range=range(n_func))
    else:
        pla1 = PersLandscapeExact(dgms=Diagrams,hom_deg=1)
        plot_landscape_simple(pla1, ax=axs[1])

    axs[1].set_title('Persistence Landscape, H1 (Dimension 1)')


    plt.tight_layout()
    plt.show()    


def noise(N1, scale):
    return scale * np.random.random((N1, 2))

def circle(N2, scale, offset):
    """Generates two circles with center at `offset` scaled by `scale`"""
    half = int(N2/2)
    circ = np.concatenate(
        (tadasets.dsphere(d=1, n=half, r=1.1, noise=0.05),
        tadasets.dsphere(d=1, n=N2-half, r=0.4, noise=0.05))
    )
    return offset + scale * circ

def generate_examples():
    np.random.seed(565656)
    M = 50           # total number of samples
    m = int(M / 2)   # number of samples per class ('noise'/'circles')
    N1 = 900          # number of points per dataset
    N2=  1000

    just_noise = [noise(N1, 150) for _ in range(m)]

    half = int(N2 / 2)
    with_circle = [np.concatenate((circle(half, 50, 70), noise(N2 - half, 150)))
                for _ in range(m)]

    datas = []
    datas.extend(just_noise)
    datas.extend(with_circle)

    # Define labels
    labels = np.zeros(M)
    labels[m:] = 1

    # Visualize the data
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(8,4)

    xs, ys = just_noise[0][:,0], just_noise[0][:,1]
    axs[0].scatter(xs, ys, s=10)
    axs[0].set_title("Example noise dataset")
    axs[0].set_aspect('equal', 'box')

    xs_, ys_ = with_circle[0][:,0], with_circle[0][:,1]
    axs[1].scatter(xs_, ys_, s=10)
    axs[1].set_title("Example noise with circle dataset")
    axs[1].set_aspect('equal', 'box')

    fig.tight_layout()

    return datas   