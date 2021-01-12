###########################################################################
# PlottingFunctions.py
# Part of HeisenbergCodes
# Updated January '21
#
# Codes for matlab plots.
# Current Qiskit:
# qiskit 0.21.0
# qiskit-terra 0.15.2
# qiskit-aer 0.6.1
# qiskit-ibmq-provider 0.9.0

# WIP
###########################################################################


import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
import numpy as np
import csv
import HelpingFunctions as hf

# TODO remove gate plots
# ==================================== Plotting Helpers ============================================================ >
colors = ['g', 'k', 'maroon', 'mediumblue', 'slateblue', 'limegreen', 'b', 'r', 'olive']
custom_cycler = (cycler(color=colors) + cycler(lw=[2] * len(colors)))
matplotlib.rcParams['figure.figsize'] = [10, 10]
plt.rc('font', size=12)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=12)


def plot_dataset_byrows(ax, legendtitle, ylabel, xlabel):
    ax.legend(loc='right', title=legendtitle)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def set_up_axes(rows, cols):
    fig, axs = plt.subplots(rows, cols)
    for col in range(cols):
        for row in range(rows):
            ax = axs[row, col]
            ax.set_prop_cycle(custom_cycler)
            ax.margins(x=0)
    fig.tight_layout(pad=3.5)
    return fig, axs


def set_up_axes_two(num_axes):
    if num_axes == 1:
        fig, ax = plt.subplots()
        ax.set_prop_cycle(custom_cycler)
        ax.margins(x=0)
        return fig, ax

    elif num_axes == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_prop_cycle(custom_cycler)
        ax2.set_prop_cycle(custom_cycler)
        ax1.margins(x=0)
        ax2.margins(x=0)
        fig.subplots_adjust(hspace=0)
        return fig, (ax1, ax2)

    elif num_axes == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.set_prop_cycle(custom_cycler)
        ax2.set_prop_cycle(custom_cycler)
        ax3.set_prop_cycle(custom_cycler)
        ax1.margins(x=0)
        ax2.margins(x=0)
        ax3.margins(x=0)
        fig.subplots_adjust(hspace=0)
        return fig, (ax1, ax2, ax3)

# =============================== Plotters for reading from files ===================================================>


def two_point_correlations_plotter_file(filename):
    vars = hf.read_var_file(filename + "_vars.csv")[0]
    n, j, total_t, dt = int(vars[0]), vars[1], int(vars[2]), vars[3]
    pairs = hf.read_var_file(filename + "_pairs.csv")[0]
    data_real_cl = hf.read_numpy_array(filename + "_cl_real.txt")
    data_imag_cl = hf.read_numpy_array(filename + "_cl_imag.txt")
    data_real_rm = hf.read_numpy_array(filename + "_q_real_RM.txt")
    data_imag_rm = hf.read_numpy_array(filename + "_q_image_RM.txt")
    data_real = hf.read_numpy_array(filename + "_q_real_nRM.txt")
    data_imag = hf.read_numpy_array(filename + "_q_imag_nRM.txt")
    re_label = r'$Re \langle S_{\alpha}^{i}(t)S_{\beta}^{j}(0)\rangle $'
    im_label = r'$Im \langle S_{\alpha}^{i}(t)S_{\beta}^{j}(0)\rangle $'
    fig, axs = set_up_axes_two(2)
    real, imag = axs[0], axs[1]
    print(dt)
    x1 = [i * abs(j) * dt for i in range(total_t)]
    print(x1)
    # have to split this into two commands since 1-d arrays are treated differently...
    if data_real_cl.ndim > 1:
        for x in pairs:
            dx = pairs.index(x)
            real.plot(x1, data_real_cl[dx][:].tolist(), label=str(x))
            real.plot(x1, data_real_rm[dx][:].tolist(), label=str(x), linestyle="dashed", linewidth=0.5)
            real.plot(x1, data_real[dx][:].tolist(), label=str(x), linestyle=":", linewidth=0.5)
            imag.plot(x1, data_imag_cl[dx][:].tolist(), label=str(x), linewidth=0.5)
            imag.plot(x1, data_imag_rm[dx][:].tolist(), label=str(x), linestyle="dashed", linewidth=0.5)
            imag.plot(x1, data_imag[dx][:].tolist(), label=str(x), linestyle=":")
    else:
        real.plot(x1, data_real_cl.tolist(), label=str(pairs[0]), linewidth=2)
        real.plot(x1, data_real_rm.tolist(), label=str(pairs[0]), linestyle="dashed", linewidth=1.5)
        real.plot(x1, data_real.tolist(), label=str(pairs[0]), linestyle=":", linewidth=1.5)
        imag.plot(x1, data_imag_cl.tolist(), label=str(pairs[0]), linewidth=2)
        imag.plot(x1, data_imag_rm.tolist(), label=str(pairs[0]), linestyle="dashed", linewidth=1.5)
        imag.plot(x1, data_imag.tolist(), label=str(pairs[0]), linestyle=":", linewidth=1.5)

    plot_dataset_byrows(real, "Site Pairs", re_label, "Jt")
    plot_dataset_byrows(imag, "Site Pairs", im_label, "Jt")
    fig.suptitle('Two-Point Correlations')
    plt.show()


def total_magnetization_plotter_file(filename):

    vars = hf.read_var_file(filename + "_vars.csv")[0]
    n, j, total_t, dt = int(vars[0]), vars[1], int(vars[2]), vars[3]
    data_cl = hf.read_numpy_array(filename + "_cl.txt")
    data_rm = hf.read_numpy_array(filename + "_q_RM.txt")
    print(data_rm)
    data = hf.read_numpy_array(filename + "_q_nRM.txt")
    fig, ax = set_up_axes_two(1)
    x1 = [i * j * dt for i in range(total_t)]
    ax.plot(x1, data_rm.tolist(), linestyle="dashed")
    ax.plot(x1, data.tolist(), linestyle=":")
    ax.plot(x1, data_cl.tolist())
    ax.set_xlabel(r'$\it{Jt}$')
    ax.set_ylabel('Total Magnetization')
    fig.suptitle('Total Magnetization')
    plt.show()


def all_site_magnetization_plotter_file(filename):

    vars = hf.read_var_file(filename + "_vars.csv")[0]
    n, j, total_t, dt = int(vars[0]), vars[1], int(vars[2]), vars[3]
    data_cl = hf.read_numpy_array(filename + "_cl.txt")
    data_two_noRM = hf.read_numpy_array(filename + "_q_nRM.txt")
    data_two_RM = hf.read_numpy_array(filename + "_q_RM.txt")
    colors_ = ['g', 'k', 'maroon', 'mediumblue', 'slateblue', 'limegreen', 'b', 'r', 'olive']
    fig, ax = set_up_axes_two(1)
    x1 = [i * abs(j) * dt for i in range(total_t)]
    sitedex = 0
    for site in range(n):
        ax.plot(x1, data_two_noRM[site][:].tolist(), linestyle="dotted", label=site)
        ax.plot(x1, data_two_RM[site][:].tolist(), linestyle="dashed", label=site)
        ax.plot(x1, data_cl[site][:].tolist(), label=site, color=colors_[sitedex])
        sitedex += 1

    plot_dataset_byrows(ax, 'Sites', 'Magnetization', r'$\it{Jt}$')
    plot_dataset_byrows(ax, 'Sites', 'Magnetization', r'$\it{Jt}$')
    fig.suptitle('Magnetization per Site', fontsize=16)
    plt.show()


#all_site_magnetization_plotter_file('tachinno_5a_Oct20')
#all_site_magnetization_plotter_file('joel_Oct20')
#total_magnetization_plotter_file('tachinno_5c_Oct20')
#two_point_correlations_plotter_file("tachinno_7a")
#two_point_correlations_plotter_file("tachinno_7b")
#two_point_correlations_plotter_file("francis")
#all_site_magnetization_plotter_file('joel_Oct27')

# ================================== Plotters for testing data runs ================================================== >

# ( functions from CompareTrotterAlgorithms)

def occ_plotter(chosen_states, j, n, total_time, dt, data, data_cl):
    data_one, data_two = data[0], data[1]
    data_one_id, data_one_noise = data_one[0], data_one[1]
    data_two_id, data_two_noise = data_two[0], data_two[1]

    fig, axs = set_up_axes(2, 2)
    x1 = [i * abs(j) * dt for i in range(total_time)]
    one, two, cl = axs[0, 0], axs[1, 0], axs[0, 1]
    for state in chosen_states:
        dex = chosen_states.index(state)
        label = np.binary_repr(state).zfill(n)
        one.plot(x1, data_one_id[dex, :].toarray().tolist()[0], label=label, linestyle="-")
        one.plot(x1, data_one_noise[dex, :].toarray().tolist()[0], label=label, linestyle=":")
        two.plot(x1, data_two_id[dex, :].toarray().tolist()[0], label=label, linestyle="-")
        two.plot(x1, data_two_noise[dex, :].toarray().tolist()[0], label=label, linestyle=":")
        cl.plot(x1, data_cl[dex, :].toarray().tolist()[0], label=label)

    plot_dataset_byrows(one, "States", "Probability", r'$\it{Jt}$')
    plot_dataset_byrows(two, "States", "Probability", r'$\it{Jt}$')
    plot_dataset_byrows(cl, "States", "Probability", r'$\it{Jt}$')
    fig.suptitle('Occupation Probabilities')
    plt.show()


def two_point_correlations_plotter(alpha, beta, j, dt, pairs, data_one, data_two, data_cl):
    # todo out alpha, beta
    real_one, imag_one = data_one[0], data_one[1]
    real_two, imag_two = data_two[0], data_two[1]
    real_one_id, imag_one_id = real_one[0], imag_one[0]
    real_one_noise, imag_one_noise = real_one[1], imag_one[1]
    real_two_id, imag_two_id = real_two[0], imag_two[0]
    real_two_noise, imag_two_noise = real_two[1], imag_two[1]
    real_cl_data, im_cl_data = data_cl[0], data_cl[1]
    re_label = r'$Re \langle S_{\alpha}^{i}(t)S_{\beta}^{j}(0)\rangle $'
    im_label = r'$Im \langle S_{\alpha}^{i}(t)S_{\beta}^{j}(0)\rangle $'

    fig, axs = set_up_axes(4, 2)
    real1, im1, real2, im2 = axs[0, 0], axs[1, 0], axs[2, 0], axs[3, 0]
    real_cl, im_cl = axs[0, 1], axs[1, 1]

    scaler = abs(j) * dt
    p = [i * scaler for i in range(len(real_one_id.toarray()[0][:].tolist()))]
    for x in pairs:
        dx = pairs.index(x)
        real1.plot(p, real_one_id.toarray()[dx][:].tolist(), label=str(x))
        real1.plot(p, real_one_noise.toarray()[dx][:].tolist(), label=str(x), linestyle=":")
        real2.plot(p, real_two_id.toarray()[dx][:].tolist(), label=str(x))
        real2.plot(p, real_two_noise.toarray()[dx][:].tolist(), label=str(x), linestyle=":")
        real_cl.plot(p, real_cl_data.toarray()[dx][:].tolist(), label=str(x))
        im_cl.plot(p, im_cl_data.toarray()[dx][:].tolist(), label=str(x))
        im1.plot(p, imag_one_id.toarray()[dx][:].tolist(), label=str(x))
        im1.plot(p, imag_one_noise.toarray()[dx][:].tolist(), label=str(x), linestyle=":")
        im2.plot(p, imag_two_id.toarray()[dx][:].tolist(), label=str(x))
        im2.plot(p, imag_two_noise.toarray()[dx][:].tolist(), label=str(x), linestyle=":")
    print('alalalla')
    plot_dataset_byrows(real1, "Site Pairs", re_label, "Jt (First-Order Trotter)")
    plot_dataset_byrows(real2, "Site Pairs", re_label, "Jt (Second-Order Trotter)")
    plot_dataset_byrows(im1, "Site Pairs", im_label, "Jt (First-Order Trotter)")
    plot_dataset_byrows(im2, "Site Pairs", im_label, "Jt (Second-Order Trotter)")
    plot_dataset_byrows(real_cl, "Site Pairs", re_label, "Jt (Exact)")
    plot_dataset_byrows(im_cl, "Site Pairs", im_label, "Jt (Exact)")

    fig.suptitle('Two-Point Correlations')
    plt.show()


def total_magnetization_plotter(j, total_t, dt, data, data_cl):
    data_one, data_two = data[0], data[1]
    data_one_id, data_one_noise = data_one[0], data_one[1]
    data_two_id, data_two_noise = data_two[0], data_two[1]

    fig, axs = set_up_axes(2, 2)
    x1 = [i * j * dt for i in range(total_t)]
    one, two, cl, gates = axs[0, 0], axs[1, 0], axs[0, 1], axs[1, 1]
    one.plot(x1, data_one_id.toarray()[0][:].tolist(), linestyle='-')
    one.plot(x1, data_one_noise.toarray()[0][:].tolist(), linestyle=":")
    two.plot(x1, data_two_id.toarray()[0][:].tolist(), linestyle='-')
    two.plot(x1, data_two_noise.toarray()[0][:].tolist(), linestyle=":")
    cl.plot(x1, data_cl.toarray()[0][:].tolist())

    one.set_xlabel(r'$\it{Jt}$')
    one.set_ylabel('Total Magnetization')
    two.set_ylabel('Total Magnetization')
    two.set_xlabel(r'$\it{Jt}$')
    cl.set_ylabel('Total Magnetization')
    cl.set_xlabel(r'$\it{Jt}$')
    fig.suptitle('Total Magnetization')
    plt.show()


def all_site_magnetization_plotter(n, j, dt, total_t, data, data_cl):
    data_one, data_two = data[0], data[1]
    data_one_id, data_one_noise = data_one[0], data_one[1]
    data_two_id, data_two_noise = data_two[0], data_two[1]
    colors_ = ['g', 'k', 'maroon', 'mediumblue', 'slateblue', 'limegreen', 'b', 'r', 'olive']

    fig, axs = set_up_axes(2, 2)
    x1 = [i * abs(j) * dt for i in range(total_t)]
    one, two, cl, gates = axs[0, 0], axs[1, 0], axs[0, 1], axs[1, 1]
    sitedex = 0
    for site in range(n):
        one.plot(x1, data_one_id.toarray()[site][:].tolist(), linestyle='-', label=site)
        one.plot(x1, data_one_noise.toarray()[site][:].tolist(), linestyle=":", label=site)
        two.plot(x1, data_two_id.toarray()[site][:].tolist(), linestyle='-', label=site)
        two.plot(x1, data_two_noise.toarray()[site][:].tolist(), linestyle=":", label=site)
        cl.plot(x1, data_cl.toarray()[site][:].tolist(), label=site, color=colors_[sitedex])
        sitedex += 1

    plot_dataset_byrows(one, 'Sites', 'Magnetization', r'$\it{Jt}$')
    plot_dataset_byrows(two, 'Sites', 'Magnetization', r'$\it{Jt}$')
    plot_dataset_byrows(cl, 'Sites', 'Magnetization', r'$\it{Jt}$')
    fig.suptitle('Magnetization per Site', fontsize=16)
    plt.show()


def set_up_axes_two_b(num_axes):
    # Used with ErrorMitigationJoel2a
    if num_axes == 1:
        fig, ax = plt.subplots()
        ax.set_prop_cycle(custom_cycler)
        ax.margins(x=0)
        return fig, ax


def all_site_magnetization_plotter_b(n, j, dt, total_t, data_one_noise, data_cl): # Might need to ditch this one, dont know. TODO
    # Used with ErrorMitigation Joel2a
    colors_ = ['g', 'k', 'maroon', 'mediumblue', 'slateblue', 'limegreen', 'b', 'r', 'olive']
    fig, ax = set_up_axes_two(1)
    x1 = [i * abs(j) * dt for i in range(total_t)]
    sitedex = 0
    for site in range(n):
        ax.plot(x1, data_one_noise.toarray()[site][:].tolist(), linestyle=":", label=site)
        ax.plot(x1, data_cl.toarray()[site][:].tolist(), label=site, color=colors_[sitedex])
        sitedex += 1
    plot_dataset_byrows(ax, 'Sites', 'Magnetization', r'$\it{Jt}$')
    fig.suptitle('Magnetization per Site', fontsize=16)
    plt.show()
