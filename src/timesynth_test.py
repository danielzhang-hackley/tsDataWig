# import timesynth as ts
import matplotlib.pyplot as plt
import datawig as dw

time_sampler = ts.TimeSampler(stop_time=20)
irregular_time_samples = time_sampler.sample_irregular_time(num_points=500, keep_percentage=50)
red_noise = ts.noise.RedNoise(std=1, tau=2)

sinusoid = ts.signals.PseudoPeriodic(frequency=1, freqSD=0.015, ampSD=0.05)
timeseries_corr = ts.TimeSeries(sinusoid, noise_generator=red_noise)
samples_corr, signals_corr, errors_corr = timeseries_corr.sample(irregular_time_samples)

car = ts.signals.CAR(ar_param=0.9, sigma=0.01)
car_series = ts.TimeSeries(signal_generator=car)


samples = car_series.sample(irregular_time_samples)
plt.plot(irregular_time_samples, samples[0], marker='o')
samples1 = car_series.sample(irregular_time_samples)
plt.plot(irregular_time_samples, samples1[0], marker='o')

plt.show()
