from mfcc import *
import numpy
import time

testfile1=[
      "music/talkings/test/test.wav",
      "music/talkings/OSR_us_000_0010_8k.wav",
      "music/talkings/OSR_us_000_0030_8k.wav",
      "music/songs/修心亭-眉间雪.wav",
      "music/songs/Pauu-追光者.wav"
      ]

testfile2=[
      "music/talkings/test/test1-list1.1.wav",
      "music/talkings/test/test1-list1.2.wav",
      "music/talkings/test/test1-list1.3.wav",
      "music/talkings/test/test1-list1.4.wav",
      "music/talkings/test/test1-list1.5.wav",
      "music/songs/Pauu-打上花火.wav",
      "music/songs/修心亭-九万字.wav",
      "music/songs/修心亭-上邪.wav",
      "music/talkings/OSR_us_000_0011_8k.wav",
      "music/talkings/OSR_us_000_0012_8k.wav",
      "music/talkings/OSR_us_000_0013_8k.wav",
      "music/talkings/OSR_us_000_0014_8k.wav",
      "music/talkings/OSR_us_000_0015_8k.wav",
      "music/talkings/OSR_us_000_0016_8k.wav",
      "music/talkings/OSR_us_000_0017_8k.wav",
      "music/talkings/OSR_us_000_0018_8k.wav",
      "music/talkings/OSR_us_000_0031_8k.wav",
      "music/talkings/OSR_us_000_0032_8k.wav",
      "music/talkings/OSR_us_000_0034_8k.wav",
      "music/talkings/OSR_us_000_0035_8k.wav"
      ]

filepath=testfile1

def match(t,w_t):
    len1= len(filepath)
    r=numpy.zeros((len1,10))
    wr=numpy.zeros((len1,10))
    orp = numpy.zeros((len1, 10))
    owrp = numpy.zeros((len1, 10))
    time1=0
    time2=0
    time3=0
    time4=0
    start=time.time()
    for i in range(0,len1):
        for j in range(0,10):
            mf = MFCC(filepath[i], 3*j)
            t1=time.time()
            r[i][j] = MFCC_R(t, mf)
            t2=time.time()
            orp[i][j] = O_dis(t, mf)
            t3=time.time()
            wmf = wei_MFCC(mf)
            t4=time.time()
            wr[i][j]=MFCC_R(w_t,wmf)
            t5=time.time()
            owrp[i][j] = O_dis(w_t, wmf)
            t6=time.time()

            time1+=(t2-t1)
            time2+=(t3-t2)
            time3+=(t5-t4)
            time4+=(t6-t5)
    maxi=numpy.max(r)
    xr=numpy.where(r==maxi)[0][0]
    omin = numpy.min(orp)
    oxr = numpy.where(orp == omin)[0][0]
    print(filepath[xr],'  耗时：',str(time1),filepath[oxr],' 耗时：',str(time2))
    maxiw=numpy.max(wr)
    wxr=numpy.where(wr==maxiw)[0][0]
    ominw = numpy.min(owrp)
    owxr = numpy.where(owrp == ominw)[0][0]
    print(filepath[wxr],'  耗时：',str(time3),filepath[owxr],' 耗时：',str(time4),"\n")

if __name__ == '__main__':
    '''len1 = len(filepath1=
    starttime= time.time()'''
    for i in range(0,len(testfile2)):
        test = MFCC(testfile2[i], 0)
        w_test = wei_MFCC(test)
        match(test, w_test)
    '''endtime = time.time()
    dtime = endtime - starttime
    print("程序运行时间：%.8s s" % dtime)  # 显示到微秒'''
