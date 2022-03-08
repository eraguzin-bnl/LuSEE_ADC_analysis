#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pandas as pd
from scipy.signal import blackman
import math

class ADC():
    def __init__(self, input_file):

        df = pd.read_csv(input_file, engine='python', usecols=[0,1])
        print(df)

        df_info=pd.read_csv(input_file, engine='python', usecols=[3,4], names=[0,1], nrows=11)
        print(df_info)
        samples = int(df_info.iat[4,1])
        sample_rate = int(df_info.iat[3,1]) * 1e6

        print(samples)
        print(sample_rate)


        # Number of sample points
        N = samples
        w = blackman(N)
        # sample spacing

        T = 1.0 / sample_rate

        x = np.linspace(0.0, N*T, N, endpoint=False)

        y = df['Magnitude'].tolist()
        print(y)
        print(type(y))
#        y = pd.DataFrame([0,1,2])
#        print(y)
#        print(type(y))

        yf = fft(y)
        ywf = fft(y*w)
        xf = fftfreq(N, T)[:N//2]

        plt.semilogy(xf, 20 * math.log10(2.0/N * np.abs(yf[0:N//2])))
        plt.semilogy(xf, 20 * math.log10(2.0/N * np.abs(ywf[0:N//2])))

        plt.grid()

        plt.show()

if __name__ == "__main__":
    ADC("ad9253_outputs/50MHz/samples.csv")
