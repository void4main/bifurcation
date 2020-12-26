import numpy as np
from numba import vectorize
import matplotlib.pyplot as plt

filename = "plot-cuda.png"

@vectorize(["float32(float32,float32,int32)"], target='cuda')
def CalcBifurcation(X,Y,iterations):
    Y = 0.1
    for i in range(1,iterations):
        Y = X * Y * (1 - Y)
    return Y

def main():
    start = 2.8
    stop = 4.0
    step = 0.0001

    X = np.arange(start, stop, step, dtype="float32")
    n = int((stop-start)/step)+1
    Y = np.zeros(n, dtype="float32")
    Y = CalcBifurcation(X,Y,150)

    plt.xlim(2.8,4)
    for i in range(1,150):
        Y = CalcBifurcation(X,Y,i)
        if i > 140:
            plt.scatter(X,Y,0.002,marker='o',cmap='hsv',alpha=0.1)
    plt.savefig(filename,dpi=800)


if __name__ == "__main__":
    main()
