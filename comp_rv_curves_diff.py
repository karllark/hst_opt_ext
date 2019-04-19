import argparse

import numpy as np
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl

from dust_extinction.parameter_averages import (CCM89, O94, F04, VCG04,
                                                GCC09, M14, F19)


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

        indxs, = np.where(np.logical_and(
            x.value >= ext_model.x_range[0],
            x.value <= ext_model.x_range[1]))
        yvals = ext_model(x[indxs])

        if plot_exvebv:
            yvals = (yvals - 1.0)*cur_Rv

        if diff_model is not None:
            dmodel = diff_model(Rv=cur_Rv)
            dvals = dmodel(x[indxs])
            if plot_exvebv:
                dvals = (dvals - 1.0)*cur_Rv
            yvals = yvals - dvals

            yoff = k*yoff_delta
        else:
            yoff = 0.0

        ax.plot(x[indxs], yvals + yoff,
                linestyle=linestyle, color=cols[k])

        if model == F19:
            ax.plot(x[indxs], yvals + yoff, 'ko',
                    markersize=5, markevery=5, markerfacecolor="None")


if __name__ == '__main__':

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt_only", help="plot the optical",
                        action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file",
                        action="store_true")
    args = parser.parse_args()

    fontsize = 12

    font = {'size': fontsize}

    mpl.rc('font', **font)

    mpl.rc('lines', linewidth=1)
    mpl.rc('axes', linewidth=2)
    mpl.rc('xtick.major', width=2)
    mpl.rc('xtick.minor', width=2)
    mpl.rc('ytick.major', width=2)
    mpl.rc('ytick.minor', width=2)

    fig, ax = plt.subplots(nrows=2, figsize=(8., 10.), sharex=True)

    plot_exvebv = True

    # generate the curves and plot them
    # x = np.arange(1.0, 3.01, 0.025)/u.micron
    if args.opt_only:
        x = np.arange(1.0, 3.3, 0.025)/u.micron
        pmodels = [CCM89, O94, F04, M14, F19]
        pmodel_names = ['CCM89', 'O94', 'F04', 'M14', 'F19']
        if plot_exvebv:
            yoff_delta = 0.4
        else:
            yoff_delta = 0.1
    else:
        x = np.arange(3.3, 8.70, 0.025)/u.micron
        pmodels = [CCM89, VCG04, F04, GCC09, F19]
        pmodel_names = ['CCM89', 'VCG04', 'F04', 'GCC09', 'F19']
        if plot_exvebv:
            yoff_delta = 2.0
        else:
            yoff_delta = 1.0

    Rvs = [2.5, 3.1, 4.5]
    cols = ['b', 'k', 'r', 'g']

    dmodel = [None, F19]

    linetypes = [':', '--', '-', '-.', '-']
    for k, cmodel in enumerate(dmodel):
        for i, cpmodel in enumerate(pmodels):
            plot_rv_set(ax[k], cpmodel, Rvs, cols, linetypes[i],
                        diff_model=cmodel,
                        plot_exvebv=plot_exvebv,
                        yoff_delta=yoff_delta)

    if plot_exvebv:
        ax[0].set_ylabel(r'$k(\lambda - 55)$')
        ax[1].set_ylabel(
            r'$k(\lambda - 55) - k(\lambda - 55)_{F19}$ + offset')
    else:
        ax[0].set_ylabel(r'$A(x)/A(V)$')
        ax[1].set_ylabel(
            r'$A(x)/A(V) - \left( A(x)/A(V) \right)_{F19}$ + offset')

    ax[1].set_xlabel(r'$x$ [$\mu m^{-1}$]')

    # legend for R(V) model type
    custom_lines = [Line2D([0], [0], color='b', linestyle='-', lw=2),
                    Line2D([0], [0], color='k', linestyle='-', lw=2),
                    Line2D([0], [0], color='r', linestyle='-', lw=2)]

    leg1 = ax[0].legend(custom_lines, ['R(V) = 2.5',
                        'R(V) = 3.1', 'R(V) = 4.5'],
                        loc='lower right')

    # legend for R(V) model type
    custom_lines = [Line2D([0], [0], color='k', linestyle=':', lw=2),
                    Line2D([0], [0], color='k', linestyle='--', lw=2),
                    Line2D([0], [0], color='k', linestyle='-', lw=2),
                    Line2D([0], [0], color='k', linestyle='-.', lw=2),
                    Line2D([0], [0], color='k', linestyle='-', lw=2,
                           marker='o', markersize=5, markerfacecolor="None")]

    ax[0].legend(custom_lines, pmodel_names,
                 loc='upper left')

    ax[0].add_artist(leg1)

    # ax.set_ylim(-3.0, 2.0)

    plt.tight_layout()

    save_file = 'comp_rv_curves_diff_uv.pdf'
    if args.opt_only:
        print('test')
        save_file.replace('uv.pdf', 'opt.pdf')
        print(save_file)
    print(save_file)
    if args.pdf:
        fig.savefig(save_file)
    else:
        plt.show()
