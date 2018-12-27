import numpy as np
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl

from dust_extinction.parameter_averages import (CCM89, O94, F04, M14)


def plot_rv_set(ax,
                model,
                rvs,
                cols,
                linestyle,
                diff_model=None,
                plot_exvebv=False,
                yoff_delta=0.4):

    for k, cur_Rv in enumerate(Rvs):
        ext_model = model(Rv=cur_Rv)
        yvals = ext_model(x)
        if plot_exvebv:
            yvals = (yvals - 1.0)*cur_Rv

        if diff_model is not None:
            dmodel = diff_model(Rv=cur_Rv)
            dvals = dmodel(x)
            if plot_exvebv:
                dvals = (dvals - 1.0)*cur_Rv
            yvals = yvals - dvals

            yoff = k*yoff_delta
        else:
            yoff = 0.0

        ax.plot(x, yvals + yoff,
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

    fig, ax = plt.subplots(nrows=2, figsize=(6., 10.), sharex=True)

    plot_exvebv = True

    # generate the curves and plot them
    x = np.arange(1.0, 3.01, 0.1)/u.micron
    Rvs = [2.5, 3.1, 4.5, 6.0]
    cols = ['b', 'k', 'r', 'g']

    dmodel = [None, M14]
    for k, cmodel in enumerate(dmodel):
        plot_rv_set(ax[k], CCM89, Rvs, cols, ':',
                    diff_model=cmodel,
                    plot_exvebv=plot_exvebv)
        plot_rv_set(ax[k], O94, Rvs, cols, '--',
                    diff_model=cmodel,
                    plot_exvebv=plot_exvebv)
        plot_rv_set(ax[k], F04, Rvs, cols, '-',
                    diff_model=cmodel,
                    plot_exvebv=plot_exvebv)
        plot_rv_set(ax[k], M14, Rvs, cols, '-.',
                    diff_model=cmodel,
                    plot_exvebv=plot_exvebv)

    if plot_exvebv:
        ax[0].set_ylabel(r'$k(\lambda - 55)$')
        ax[1].set_ylabel(r'$k(\lambda - 55) - k(\lambda - 55)_{M14}$')
    else:
        ax[0].set_ylabel(r'$A(x)/A(V)$')
        ax[1].set_ylabel(r'$A(x)/A(V) - \left( A(x)/A(V) \right)_{M14}$')

    ax[1].set_xlabel(r'$x$ [$\mu m^{-1}$]')

    # legend for R(V) model type
    custom_lines = [Line2D([0], [0], color='g', linestyle='-', lw=2),
                    Line2D([0], [0], color='r', linestyle='-', lw=2),
                    Line2D([0], [0], color='k', linestyle='-', lw=2),
                    Line2D([0], [0], color='b', linestyle='-', lw=2)]

    leg1 = ax[0].legend(custom_lines, ['R(V) = 6.0', 'R(V) = 4.5',
                        'R(V) = 3.1', 'R(V) = 2.5'],
                        loc='lower right')

    # legend for R(V) model type
    custom_lines = [Line2D([0], [0], color='k', linestyle=':', lw=2),
                    Line2D([0], [0], color='k', linestyle='--', lw=2),
                    Line2D([0], [0], color='k', linestyle='-', lw=2),
                    Line2D([0], [0], color='k', linestyle='-.', lw=2)]

    ax[0].legend(custom_lines, ['CCM89', 'O94', 'F04', 'M14'],
                 loc='upper left')

    ax[0].add_artist(leg1)

    # ax.set_ylim(-3.0, 2.0)

    plt.tight_layout()

    plt.show()
