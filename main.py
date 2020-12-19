import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mfcc import *
import numpy
filepath1=["music/songs/unravel.wav",
          "music/songs/大鱼-周深.wav",
          "music/talkings/OSR_us_000_0011_8k.wav",
          "music/talkings/OSR_us_000_0012_8k.wav",
          "music/talkings/OSR_us_000_0013_8k.wav",
          "music/talkings/OSR_us_000_0014_8k.wav",]

def match(t,w_t):
    len1= len(filepath1)
    r=numpy.zeros((len1,10))
    wr=numpy.zeros((len1,10))
    for i in range(0,len1):
        for j in range(0,10):
            mf = MFCC(filepath1[i], 3*j)
            r[i][j] = MFCC_R(t, mf)
            wmf = wei_MFCC(mf)
            wr[i][j] = MFCC_R(w_t, wmf)
    maxi=numpy.max(r)
    xr=numpy.where(r==maxi)[0][0]
    yr=3*numpy.where(r==maxi)[1][0]
    print(filepath1[xr],yr,'s起')
    maxiw=numpy.max(wr)
    wxr=numpy.where(wr==maxiw)[0][0]
    wyr=3*numpy.where(wr==maxiw)[1][0]
    print(filepath1[wxr],wyr,'s起')

if __name__ == '__main__':
    test=MFCC("music/songs/unravel.wav",3)
    w_test=wei_MFCC(test)
    match(test,w_test)
    '''plt.plot(wr,r,'go-', label='aa')
    plt.xlabel('this is r')
    plt.ylabel('this is wr')
    plt.title('this is a demo')
    plt.legend()  # 将样例显示出来

    plt.plot()
    plt.show()'''
