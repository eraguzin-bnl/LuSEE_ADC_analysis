#!/usr/bin/env python3
import os.path

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pandas as pd
from scipy.signal.windows import blackman
from matplotlib.ticker import FormatStrFormatter
import yaml
import sys
import math


class ADC:
    def __init__(self):
        self.config = None
        self.samples = None
        self.sample_rate = None
        self.data = None
        self.prev_fft_frequency = None
        self.prev_fft_amplitude = None
        self.subtitle_font_size = 36
        self.label_size = 28
        self.tick_size = 22
        self.legend_size = 24

    def plot(self):
        # Number of sample points
        N = self.samples
        w = blackman(N)
        # sample spacing

        T = 1.0 / self.sample_rate

        x = np.linspace(0.0, N * T, N, endpoint=False) * 1e9

        yf = fft(self.data)
        ywf = fft(self.data * w)
        xf = fftfreq(N, T)[:N // 2]

        yplot = 2.0 / N * np.abs(yf[0:N // 2])
        ywplot = 2.0 / N * np.abs(ywf[0:N // 2])

        if self.config["include_samples"]:
            fig, (sample_ax, fft_ax) = plt.subplots(nrows=2, ncols=1, figsize=(32, 24))
            sample_ax.plot(x[0:self.config["num_samples"]], self.data[0:self.config["num_samples"]])
            # sample_ax.set_xlim([0.00004, 0.000041])
            fft_ax.xaxis.set_major_formatter(FormatStrFormatter('%10.f'))
            sample_ax.set_xlabel("Time (ns)", fontsize=self.label_size)
            fft_ax.set_xlabel("ADC Codes", fontsize=self.label_size)
            sample_ax.set_title(f"{self.config['name']} samples", fontsize=self.subtitle_font_size)
            sample_ax.tick_params(axis='x', labelsize=self.tick_size)
            sample_ax.tick_params(axis='y', labelsize=self.tick_size)
        else:
            fig, fft_ax = plt.subplots(nrows=1, ncols=1, figsize=(32, 24))

        fft_ax.plot(xf / 1e6, 20 * np.log10(yplot / max(yplot)), '-b')
        fft_ax.plot(xf / 1e6, 20 * np.log10(ywplot / max(ywplot)), '-r')

        if self.config["include_previous"]:
            fft_ax.plot(self.prev_fft_frequency, self.prev_fft_amplitude, 'black', alpha=1, linestyle='dashed')

        fft_ax.grid()
        plt.subplots_adjust(wspace=0, hspace=0.15, top=0.95, bottom=0.05, right=0.92, left=0.09)

        fft_ax.set_ylabel("Amplitude (dBFS)", fontsize=self.label_size)
        fft_ax.set_xlabel("Frequency (MHz)", fontsize=self.label_size)
        fft_ax.xaxis.set_major_formatter(FormatStrFormatter('%10.f'))

        fft_ax.tick_params(axis='x', labelsize=self.tick_size)
        fft_ax.tick_params(axis='y', labelsize=self.tick_size)

        fft_ax.legend(['FFT(no window)', 'FFT(Blackman window)', 'From software'], fontsize=self.legend_size)
        fft_ax.set_title(f"{self.config['name']} Fast Fourier Transform", fontsize=self.subtitle_font_size)

        output_dir = os.path.join(os.getcwd(), "plot_outputs")
        if not (os.path.isdir(output_dir)):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, self.config["plot_file"]), bbox_inches="tight")
        # plt.show()


class AD9253(ADC):
    def __init__(self, config_in):
        super().__init__()
        self.config = config_in
        filepath = os.path.join(config_in["data_dir"], config_in["data_file"])
        df = pd.read_csv(filepath, engine='python', usecols=[0, 1])
        # Needs to be grabbed this way and indexed because use_cols is being deprecated
        df_info_filler = pd.read_csv(filepath, engine='python', names=["Filler1", "Filler2", "Filler3",
                                                                       "Info_Name", "Info_Value", "Units"], nrows=11)
        df_info = pd.DataFrame({"name": df_info_filler["Info_Name"], "value": df_info_filler["Info_Value"],
                                "units": df_info_filler["Units"]})

        self.samples = int(df_info.iat[4, 1])
        sample_rate_unit = df_info.iat[3, 2]
        if sample_rate_unit == "MHz":
            sample_rate_multiplier = 1e6
        else:
            raise Exception(f"Unknown sample unit. Sample unit is {sample_rate_unit}")
        self.sample_rate = int(df_info.iat[3, 1]) * sample_rate_multiplier
        self.data = df['Magnitude'].tolist()

        if config_in["include_previous"]:
            filepath = os.path.join(config_in["data_dir"], config_in["previous_file"])
            df2 = pd.read_csv(filepath, engine='python', usecols=[0, 1])
            self.prev_fft_frequency = (df2['Frequency'] / 1e6).tolist()
            self.prev_fft_amplitude = df2['Amplitude'].tolist()

        self.plot()


class ADC3664(ADC):
    def __init__(self, config_in):
        super().__init__()
        self.config = config_in
        filepath = os.path.join(config_in["data_dir"], config_in["data_file"])
        df = pd.read_excel(filepath, usecols=[2, 3], names=["Sample", "Data"])
        self.sample_rate = int(config_in["input_sample_rate"]) * 1e6
        self.data = df['Data'].tolist()
        self.samples = len(self.data)
        if config_in["include_previous"]:
            filepath = os.path.join(config_in["data_dir"], config_in["previous_file"])
            df2 = pd.read_excel(filepath, usecols=[0, 1], names=["Frequency", "Amplitude"])
            self.prev_fft_frequency = (df2['Frequency'] / 1e6).tolist()
            self.prev_fft_amplitude = df2['Amplitude'].tolist()

        self.plot()


class ADS4245(ADC3664):
    def __init__(self, config_in):
        super().__init__(config_in)


if __name__ == "__main__":
    known_adcs = {"AD9253": AD9253, "ADC3664": ADC3664, "ADS4245": ADS4245}
    with open("lusee_analysis.yaml", "r") as f:
        config = yaml.safe_load(f)

    for i in config:
        print(f"Running analysis {i}")
        adc_type = config[i]['adc']
        if adc_type in known_adcs:
            adc_object = known_adcs[adc_type]
            adc_object(config[i])
        else:
            raise f"ADC type {adc_type} is unknown"