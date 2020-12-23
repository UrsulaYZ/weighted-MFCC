import numpy
import numpy.linalg as lg
import scipy.io.wavfile
import math
from matplotlib import pyplot as plt
from scipy.fftpack import dct

def MFCC(filepath,begin):
    if len(filepath)==0:
        return
    sample_rate, signal = scipy.io.wavfile.read(filepath)

    #print(sample_rate, len(signal))
    # 读取前3s 的数据
    signal = signal[begin*sample_rate:int((begin+3) * sample_rate)]
    #print(signal)

    # 预先处理
    pre_emphasis = 0.98
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length+frame_step)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z)

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
        numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[numpy.mat(indices).astype(numpy.int32, copy=False)]

    # 加上汉明窗
    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

    # 傅立叶变换和功率谱
    NFFT = 512
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    # print(mag_frames.shape)
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # Power Spectrum

    low_freq_mel = 0
    # 将频率转换为Mel
    nfilt = 40
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz

    bins = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bins[m - 1])  # left
        f_m = int(bins[m])  # center
        f_m_plus = int(bins[m + 1])  # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape

    n = numpy.arange(ncoeff)
    cep_lifter = 22
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift  # *

    # filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
    '''plt.plot(filter_banks)
    plt.show()'''
    return mfcc

def MFCC_sta(m):
    row=m.shape[0]
    col=m.shape[1]
    maxi=numpy.zeros((1,col))
    mini=numpy.zeros((1,col))
    s=numpy.copy(m)
    for j in range(0,col):
        for i in range(0,row):
            if i==0:
                maxi[0][j]=s[0][j]
                mini[0][j]=s[0][j]
            else:
                if s[i][j]-maxi[0][j]>0:
                    maxi[0][j]=s[i][j]
                elif mini[0][j]-s[i][j]>0:
                    mini[0][j]=s[i][j]
    for i in range(0,row):
        for j in range(0,col):
            if maxi[0][j]-mini[0][j]!=0:
               s[i][j]=(maxi[0][j]-s[i][j])/(maxi[0][j]-mini[0][j])
            elif maxi[0][j]-mini[0][j]==maxi[0][j]-s[i][j]:#都是0
                #print("Warning:no voice")
                s[i][j]=1
            else:#有一个是0
                #print("Warning:no voice")
                s[i][j]=0

    return s


def weight(sta):
    row=sta.shape[0]
    col=sta.shape[1]
    colsum=numpy.zeros((1,col))
    Y=numpy.zeros((row,col))
    for j in range(0,col):
        for i in range(0,row):
            colsum[0][j]+=sta[i][j]
    for i in range(0,row):
        for j in range(0,col):
            Y[i][j]=sta[i][j]/colsum[0][j]
    e=numpy.zeros((1,col))
    sume=0
    for j in range(0,col):
        for i in range(0,row):
            if Y[i][j]==0:
                continue
            e[0][j]-=Y[i][j]*math.log(Y[i][j],math.e)#每一列的该结果求和
            #e[0][j]*=k
        sume+=e[0][j]  #sume是e[j]的和
    w=numpy.ndarray(shape=[1,col])
    for j in range(0,col):
        w[0][j]=(1-e[0][j])/(col-sume)
    '''arr=w.flatten()
    plt.bar(numpy.arange(col), arr,width=0.5)
    plt.show()'''
    return w

def MFCC_R(mfcc1,mfcc2):
    x = numpy.dot(mfcc1, mfcc2.T)
    value1=numpy.linalg.norm(x)
    value2=numpy.linalg.norm(numpy.dot(mfcc1,mfcc1.T))
    value3=numpy.linalg.norm(numpy.dot(mfcc2,mfcc2.T))
    y=math.sqrt(value2*value3)
    res=value1/y
    return res

def O_dis(mfcc1,mfcc2):
    op = numpy.linalg.norm(mfcc1 - mfcc2)
    return op

def wei_MFCC(mfcc):
    STA=MFCC_sta(mfcc)
    w=weight(STA)
    res=numpy.multiply(mfcc,w)
    return res
