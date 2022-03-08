#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pandas as pd
from scipy.signal import blackman
import math
from matplotlib.ticker import FormatStrFormatter

class ADC():
    def __init__(self, input_file):

        df = pd.read_csv(input_file, engine='python', usecols=[0,1])
#        print(df)

        df_info=pd.read_csv(input_file, engine='python', usecols=[3,4], names=[0,1], nrows=11)
#        print(df_info)
        samples = int(df_info.iat[4,1])
        sample_rate = int(df_info.iat[3,1]) * 1e6

#        print(samples)
#        print(sample_rate)


        # Number of sample points
        N = samples
        w = blackman(N)
        # sample spacing

        T = 1.0 / sample_rate

        x = np.linspace(0.0, N*T, N, endpoint=False)

        y = df['Magnitude'].tolist()
#        print(y)
#        print(type(y))
#        y = pd.DataFrame([0,1,2])
#        print(y)
#        print(type(y))

        yf = fft(y)
        ywf = fft(y*w)
        xf = fftfreq(N, T)[:N//2]

        yplot = 2.0/N * np.abs(yf[0:N//2])
        ywplot = 2.0/N * np.abs(ywf[0:N//2])

        fig, (sample_ax, fft_ax) = plt.subplots(nrows=2,ncols=1,figsize=(32,24))
        fig.suptitle("AD9253 50MHz signal analysis")
        #plt.subplots_adjust(wspace=0,hspace=0,top=0.95,bottom=0.05,right=0.92,left=0.09)

        sample_ax.plot(x, y)
        sample_ax.set_xlim([0.00004, 0.000041])
        sample_ax.set_xlabel("Time (s)")

        fft_ax.plot(xf/1e6, 20 * np.log10(yplot/max(yplot)), '-b')
        fft_ax.plot(xf/1e6, 20 * np.log10(ywplot/max(ywplot)), '-r')
        df2 = pd.read_csv('ad9253_outputs/50MHz/fft2.csv', engine='python', usecols=[0,1])
        x = (df2['Frequency']/1e6).tolist()
        y = df2['Amplitude'].tolist()
        fft_ax.plot(x,y, 'black', alpha=1, linestyle='dashed')

        fft_ax.grid()

        fft_ax.set_ylabel("Amplitude (dBFS)")
        fft_ax.set_xlabel("Frequency (MHz)")
        fft_ax.xaxis.set_major_formatter(FormatStrFormatter('%10.f'))




        fft_ax.legend(['FFT(no window)', 'FFT(Blackman window)', 'From software'])

        plt.show()

if __name__ == "__main__":
    ADC("ad9253_outputs/50MHz/samples.csv")
