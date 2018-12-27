import numpy as np
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl

from dust_extinction.parameter_averages import (CCM89, O94, F04, M14)


def plot_rv_set(model,
                rvs,
                cols,
                linestyle,
                plot_exvebv=False):

    for k, cur_Rv in enumerate(Rvs):
        ext_model = model(Rv=cur_Rv)
        yvals = ext_model(x)
        if plot_exvebv:
            yvals = (yvals - 1.0)*cur_Rv
        ax.plot(x, yvals,
                linestyle=linestyle, color=cols[k])


if __name__ == '__main__':

    fontsize = 12

    font = {'size': fontsize}

    mpl.rc('font', **font)

    mpl.rc('lines', linewidth=1)
    mpl.rc('axes', linewidth=2)
    mpl.rc('xtick.major', width=2)
    mpl.rc('xtick.minor', width=2)
    mpl.rc('ytick.major', width=2)
    mpl.rc('ytick.minor', width=2)

    fig, ax = plt.subplots(figsize=(10., 6.))

    plot_exvebv = False

    # generate the curves and plot them
    x = np.arange(1.0, 3.0, 0.1)/u.micron
    Rvs = [2.5, 3.1, 4.5, 6.0]
    cols = ['b', 'k', 'r', 'g']

    plot_rv_set(CCM89, Rvs, cols, ':', plot_exvebv=plot_exvebv)

    plot_rv_set(O94, Rvs, cols, '--', plot_exvebv=plot_exvebv)

    plot_rv_set(F04, Rvs, cols, '-', plot_exvebv=plot_exvebv)

    plot_rv_set(M14, Rvs, cols, '-.', plot_exvebv=plot_exvebv)

    ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
    ax.set_ylabel(r'$A(x)/A(V)$')

    # legend for R(V) model type
    custom_lines = [Line2D([0], [0], color='b', linestyle='-', lw=2),
                    Line2D([0], [0], color='k', linestyle='-', lw=2),
                    Line2D([0], [0], color='r', linestyle='-', lw=2),
                    Line2D([0], [0], color='g', linestyle='-', lw=2)]

    leg1 = ax.legend(custom_lines, ['R(V) = 2.5', 'R(V) = 3.1',
                                    'R(V) = 4.5', 'R(V) = 6.0'],
                     loc='lower right')

    # legend for R(V) model type
    custom_lines = [Line2D([0], [0], color='k', linestyle=':', lw=2),
                    Line2D([0], [0], color='k', linestyle='--', lw=2),
                    Line2D([0], [0], color='k', linestyle='-', lw=2),
                    Line2D([0], [0], color='k', linestyle='-.', lw=2)]

    ax.legend(custom_lines, ['CCM89', 'O94', 'F04', 'M14'],
              loc='upper left')

    ax.add_artist(leg1)

    # ax.set_ylim(-3.0, 2.0)

    plt.tight_layout()

    plt.show()
