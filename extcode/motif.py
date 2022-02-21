from math import log
import numpy as np
import math
import matplotlib.pyplot as plt
import pdb


def despine(ax,right=True, top=True, left=False,bottom=False):
    if left:
        ax.spines['left'].set_visible(False)
    if bottom:
        ax.spines['bottom'].set_visible(False)

    if right:
        ax.spines['right'].set_visible(False)
    if top:
        ax.spines['top'].set_visible(False)

def draw_A(width, height,x,y, ax, color="blue",lw=2.0,alpha=0.5):
    ax.plot([x, x+width*0.5 ], [ y, y+height ],lw=lw, color=color ,alpha=alpha)
    ax.plot([ x+width*0.5, x+width ], [ y+height, y ],lw=lw, color=color ,alpha=alpha)
    ax.plot([x+width*0.25,x+width*0.75], [height*0.5, height*0.5], color=color, lw=lw,alpha=alpha)
    return y+height
def draw_T(width, height,x,y, ax, color="red",lw=2.0,alpha=0.5):
    center  = width*0.5
    ax.plot([x+center, x+center], [y, y+height],lw=lw, color=color,alpha=alpha)
    ax.plot([x, x+width], [y+height, y+height],lw=lw, color=color,alpha=alpha)
    return y+height
def draw_C(width, height,x,y, ax, color="green",lw=2.0,alpha=0.5):
    width/=2.0
    x       += width
    y       -=math.sin(3*math.pi/2.)*height
    ts      = np.linspace(3.85*math.pi/2., math.pi/6., 100)
    xs      = [ math.cos(t)*width+x for t in ts]
    ys      = [ math.sin(t)*height+y for t in ts]
    ax.plot(xs,ys,lw=lw, color=color,alpha=alpha)

    return y + math.sin(math.pi/2.)*height 
def draw_G(width, height,x,y, ax, color="orange",lw=2.0,alpha=0.5):
    width/=2.0
    x       += width

    y       += math.sin(math.pi/2.)*height
    ts      = np.linspace(3.95*math.pi/2., math.pi/6., 100)
    xs      = [ math.cos(t)*width+x for t in ts]
    ys      = [ math.sin(t)*height+y for t in ts]

    ax.plot(xs,ys,lw=lw, color=color,alpha=alpha)
    ptl     = math.cos(3*math.pi/2.)*width+x,  math.sin(3*math.pi/2.)*height+y

    ptu     = xs[0],ys[0]

    ax.plot([ ptu[0],ptu[0] ], [ptu[1], ptl[1] ],lw=lw, color=color,alpha=alpha)
    ax.plot([ ptu[0]-width*0.75,ptu[0] ], [ptu[1], ptu[1] ],lw=lw, color=color,alpha=alpha)
    return y + math.sin(math.pi/2.)*height

def draw_seq_log(PSSM,title, AX=None,lw=3.0,alpha=1):
    ax      = AX

    if AX is None:
        F   = plt.figure(facecolor="white")
        ax  = plt.gca()
    ax.set_title(title, fontsize=20)
    A       = np.array(PSSM)
    for i in range(A.shape[0]):
        A[i,:]     /= sum(A[i,:])
        methods     = (draw_A, draw_C, draw_G, draw_T)
        x           = i
        width       = 0.9
        y           = 0.0
        I           = 2+sum([A[i,l]*math.log(A[i,l],2)  for l in range(4)])
        for j in range(4):
            d       = methods[j]
            h       = I*A[i,j] 
            if j == 1 or j ==2:
                h/=2.0
            y       = d(width, h , x,y,ax,lw=lw,alpha=alpha)
    ax.set_xlim(0,A.shape[0])
    ax.set_xticks([])
    ax.set_xlabel("Position",fontsize=20)
    ax.set_ylabel("bits",fontsize=20)
    despine(ax,bottom=True)
    ax.yaxis.grid(True)
    ax.set_ylim(0,2)
    if AX is None:
        plt.show()
class logo:
    def __init__(self, matrix, name=""):
        self.m      = matrix
        self.name   = name
    def draw(self ,ax=None):
        draw_seq_log(self.m,self.name, AX=ax)


def main():
    A=[[ 0.356611,0.085184,0.455087,0.103118],
       [ 0.217352,0.033832,0.616781,0.132036],
       [ 0.411544,0.053398,0.486295,0.048762],
       [ 0.026776,0.902577,0.020027,0.050619],
       [ 0.773107,0.016504,0.055813,0.154576],
       [ 0.121956,0.027309,0.046496,0.804239],
       [ 0.010707,0.015609,0.947804,0.025881],
       [ 0.031773,0.310079,0.022036,0.636112],
       [ 0.055142,0.622140,0.063964,0.258755],
       [ 0.176130,0.035488,0.131170,0.657211],
       [ 0.095022,0.092511,0.793152,0.019314],
       [ 0.190134,0.058570,0.593589,0.157707],
       [ 0.456692,0.015034,0.473979,0.054296],
       [ 0.008153,0.976192,0.001984,0.013671],
       [ 0.767356,0.030153,0.037623,0.164869],
       [ 0.170081,0.032247,0.065448,0.732224],
       [ 0.036761,0.017177,0.926087,0.019975],
       [ 0.059032,0.416386,0.015859,0.508723],
       [ 0.114088,0.584113,0.060812,0.240987],
       [ 0.224960,0.355487,0.071233,0.348320]]


    L        = logo(A, name="P53")
    F        = plt.figure(figsize=(15,6),facecolor="white")
    ax       = plt.gca()
    L.draw(ax=ax)
    plt.show()

if __name__ == "__main__":
    main()




        