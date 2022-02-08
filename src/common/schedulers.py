import math


def cosine(epoch, channel_tau, tau_scheduler_rate, **kwargs):
    tau = channel_tau * (math.cos(math.pi * epoch * tau_scheduler_rate) + 1) / 2
    return tau


def linear(epoch, channel_tau, tau_scheduler_rate, **kwargs):
    return max(0, channel_tau - (tau_scheduler_rate * epoch))


def exponential(epoch, channel_tau, tau_scheduler_rate, **kwargs):
    if epoch == 0:
        return channel_tau
    return channel_tau * (1-(tau_scheduler_rate ** (1/epoch)))


if __name__ == '__main__':
    import matplotlib.pylab as plt
    import numpy as np

    xs = np.linspace(0, 100, num=100)
    y = [linear(x, 10, 0.1) for x in xs]
    plt.plot(xs, y, label='Linear')

    y = [exponential(x, 10, 0.1) for x in xs]
    plt.plot(xs, y, label='Exponential')

    y = [cosine(x, 10, 0.01) for x in xs]
    plt.plot(xs, y, label='cosine')

    plt.legend()
    plt.show()
