#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import gwpy
import bilby
import sys
import os
import gwpy.timeseries
import gwpy
from gwpy.timeseries import TimeSeries
from pygwb.detector import Interferometer
from pygwb.network import Network
from pygwb.baseline import Baseline
from pygwb.simulator import Simulator
from bilby.gw.detector import PowerSpectralDensity

def PL_GW(freqs, omegaref, alpha, fref):
    from pygwb.constants import H0
    H_theor = (3 * H0.si.value ** 2) / (10 * np.pi ** 2)
    
    power = np.zeros_like(freqs)
    
    power = H_theor * omegaref * freqs ** (alpha -3) / fref**(alpha)
    
    power[0] = power[1]
    
    return gwpy.frequencyseries.FrequencySeries(power, frequencies=freqs)


jobNumber = int(sys.argv[1])
st = (jobNumber-1)*2048
duration = 4096
minimum_frequency  = 5
sampling_frequency = 2048

omegaref = 5e-12
fref = 25
alpha = 0
    
interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1'])
for ifo in interferometers:
    ifo.minimum_frequency  = minimum_frequency
    ifo.sampling_frequency = sampling_frequency
    ifo.duration = duration

interferometers.set_strain_data_from_zero_noise(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=st
)
base_channel_name = 'TEST_INJ'
gps_start_time = ifo.time_array[0]

net_HL = Network('HL', interferometers)

frequencies_x = np.linspace(5, 2048, 65537) # delta_f = 1/32 Hz


Intensity_GW_inject = PL_GW(frequencies_x, omegaref, alpha,fref)


net_HL.set_interferometer_data_from_simulator(start_time=st, N_segments=1, GWB_intensity=Intensity_GW_inject, sampling_frequency=sampling_frequency, inject_into_data_flag=True)


H_data = net_HL.interferometers[0].strain_data.to_gwpy_timeseries()[1024*2048: 1024*2048+2048*2048]
L_data = net_HL.interferometers[1].strain_data.to_gwpy_timeseries()[1024*2048: 1024*2048+2048*2048]

H_data.times = np.arange(st, st+2048+1/2048, 1/2048)
L_data.times = np.arange(st, st+2048+1/2048, 1/2048)

H_data.name   = "H1:TEST_INJ"
L_data.name   = "L1:TEST_INJ"


H_Noise = TimeSeries.read(f'.../Pure_Noise/H-STRAIN-{st}-2048.gwf', "H1:TEST_INJ", st, st+2048)
L_Noise = TimeSeries.read(f'.../Pure_Noise/L-STRAIN-{st}-2048.gwf', "L1:TEST_INJ", st, st+2048)

H_data = H_data + H_Noise
L_data = L_data + L_Noise

H_data.write(f'.../Noisy_SGWB/alpha_0_Omega_5e-12/H-STRAIN-{st}-2048.gwf')
L_data.write(f'.../Noisy_SGWB/alpha_0_Omega_5e-12/L-STRAIN-{st}-2048.gwf')