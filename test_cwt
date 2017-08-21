import matplotlib.pyplot as plt
import cwt


fs = 1e3
t = np.linspace(0, 1, fs+1, endpoint=True)
x = np.cos(2*np.pi*32*t) * np.logical_and(t >= 0.1, t < 0.3) + np.sin(2*np.pi*64*t) * (t > 0.7)
wgnNoise = 0.05 * np.random.standard_normal(t.shape)
x += wgnNoise
c, f = cwt.cwt(x, 'morl', sampling_frequency=fs)

fig, ax = plt.subplots()
ax.imshow(np.absolute(wt), aspect='auto')
