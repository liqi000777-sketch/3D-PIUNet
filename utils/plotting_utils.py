import matplotlib.pyplot as plt

import matplotlib.animation as animation
import numpy as np
import pandas as pd
import re

#plt.rcParams["text.usetex"] = True


def init_plotting():
    # plt.rcParams['figure.figsize'] = (15,5)
    plt.rcParams["font.size"] = 15
    #plt.rcParams["font.family"] = "serif"  #'Times New Roman'
    #plt.rcParams["text.usetex"] = True

    #plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["axes.labelsize"] = plt.rcParams["font.size"]
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlesize"] = plt.rcParams["font.size"]
    plt.rcParams["legend.fontsize"] = plt.rcParams["font.size"]
    plt.rcParams["xtick.labelsize"] = plt.rcParams["font.size"]
    plt.rcParams["ytick.labelsize"] = plt.rcParams["font.size"]
    # plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
    #plt.rcParams["xtick.major.size"] = 3
    #plt.rcParams["xtick.minor.size"] = 3
    #plt.rcParams["xtick.major.width"] = 1
    #plt.rcParams["xtick.minor.width"] = 1
    #plt.rcParams["ytick.major.size"] = 3
    #plt.rcParams["ytick.minor.size"] = 3
    #plt.rcParams["ytick.major.width"] = 1
    #plt.rcParams["ytick.minor.width"] = 1
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.loc"] = "best"
    #plt.rcParams["axes.linewidth"] = 1
    # plt.rcParams['figure.subplot.wspace'] = 0.25
    # plt.rcParams['figure.subplot.right'] = 0.95
    plt.rcParams["legend.columnspacing"] = 1
    # plt.rcParams['legend.handleheight'] = 4*plt.rcParams['legend.handleheight']
    plt.gca().spines["right"].set_color("none")
    plt.gca().spines["top"].set_color("none")
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("default")


init_plotting()

from matplotlib.colors import LogNorm
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

MARKERS_LINE = ["-o", "-^", "-*", "-X", "-D", "-H", "-P", "-v", "-<", "->", "-1", "-s", "-d"] * 3
MARKERS = ["o", "^", "*", "X", "D", "H", "P", "v", "<", ">", "1", "s", "d"] * 3



def plot_final_prediction3(pred_x, pseudo_x, target_x, filename):
    try:
        plot_animation_volumes(
            [
                [
                    pred_x[0, 0].detach().cpu().numpy(),
                    pred_x[1, 0].detach().cpu().numpy(),
                    pred_x[2, 0].detach().cpu().numpy(),
                ],
                [
                    pseudo_x[0, 0].detach().cpu().numpy(),
                    pseudo_x[1, 0].detach().cpu().numpy(),
                    pseudo_x[2, 0].detach().cpu().numpy(),
                ],
                [
                    target_x[0, 0].detach().cpu().numpy(),
                    target_x[1, 0].detach().cpu().numpy(),
                    target_x[2, 0].detach().cpu().numpy(),
                ],
            ],
            filename=filename,
            names=["Sample 1", "Sample 2", "Sample 3"],
            y_names=[
                "Predicted",
                "Pseudo Inv",
                "GT",
            ],
        )
    except Exception as e:
        print("Could not plot final predictions", e)

def plot_final_prediction(pred_x, pseudo_x, target_x, filename):
    try:
        plot_animation_volumes(
            [
                    target_x[0, 0].detach().cpu().numpy(),
                    pseudo_x[0, 0].detach().cpu().numpy(),
                    pred_x[0, 0].detach().cpu().numpy(),

            ],
            filename=filename,
            names=[
                "GT",
                "Pseudo Inv",
                "Predicted",
            ],
        )
    except Exception as e:
        print("Could not plot final predictions", e)


def plot_pred_chain(denoised_chain, filename):
    try:
        timesteps = len(denoised_chain)
        steps = np.linspace(0, timesteps - 1, 20, dtype=int)
        plot_animation_volumes(
            [
                [denoised_chain[i][0, 0].detach().cpu().numpy() for i in steps[:10]],
                [denoised_chain[i][0, 0].detach().cpu().numpy() for i in steps[10:]],
            ],
            filename=filename,
            names=[f"{i}" for i in steps[:10]],
        )
    except Exception as e:
        print("Could not plot Predicted Chain", e)


def plot_noised_denoise_chain(denoised_chain, noised_chain, filename):
    try:
        slices = [4, 8, 12, 16]
        timesteps = len(denoised_chain)
        steps = np.linspace(0, timesteps - 1, 40, dtype=int)
        plot_animation_volumes(
            [
                [np.asarray([noised_chain[i][0, 0, slice].detach().cpu().numpy() for i in steps]) for slice in slices],
                [
                    np.asarray([denoised_chain[i][0, 0, slice].detach().cpu().numpy() for i in steps])
                    for slice in slices
                ],
            ],
            filename=filename,
            names=[f"Slice {i}" for i in slices],
            y_names=["GT", "Pred"],
        )
    except Exception as e:
        print("Could not plot noised-denoised Chain", e)


def plot_animation_volumes(volumes, filename=None, names=None, y_names=None):
    fps = 4
    if isinstance(volumes[0], list):
        # We have a 2D plot!
        nSeconds = volumes[0][0].shape[0] // fps
        n_rows = len(volumes)
        n_cols = len(volumes[0])

        volumes = np.asarray(volumes)
    else:
        nSeconds = volumes[0].shape[0] // fps
        n_rows = 1
        n_cols = len(volumes)
        volumes = np.asarray(volumes)[None]
    # First set up the figure, the axis, and the plot element we want to animate
    # fig = plt.figure(figsize=(8,8))
    vmin = np.min(volumes)  # [np.min(v) for v in volumes])
    vmax = np.max(volumes)  # [np.max(v) for v in volumes])
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 1 + 4 * n_rows))
    if n_rows == 1:
        ax = ax[None]
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=None, hspace=None)
    imgs = []
    for i in range(n_rows):
        img_row = []
        if y_names is not None:
            ax[i, 0].set_ylabel(y_names[i])
        for j, volume in enumerate(volumes[i]):
            a = volume[0]
            im = ax[i, j].imshow(a, interpolation="none", aspect="auto", vmin=vmin, vmax=vmax)
            if names is not None and i == 0:
                ax[i, j].set_title(names[j])
            img_row.append(im)
        imgs.append(img_row)

    # shared colorbar

    # fig.subplots_adjust(wspace=None, hspace=None)
    fig.colorbar(im, ax=ax.ravel().tolist(), location="bottom")
    for ax_ in ax.flatten():

        plt.setp(ax_.get_xticklabels(), visible=False)
        plt.setp(ax_.get_yticklabels(), visible=False)

    def animate_func(t):
        for i, img_row in enumerate(imgs):
            for j, img in enumerate(img_row):
                img.set_array(volumes[i, j, t])
        return imgs

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=nSeconds * fps,
        interval=1000 / fps,  # in ms
    )
    if filename:
        #TODO not sure if gif is better
        writergif = animation.PillowWriter(fps=fps)
        anim.save(filename.replace(".mp4", ".gif"), writer=writergif)
        #anim.save(filename, fps=fps, extra_args=["-vcodec", "libx264"])
    # anim.show()
    plt.close("all")
    # return anim
    # return anim


def plot_mnist_overview(imgs, filename=None, y_names=None, names=None):
    """
    We plot Blurred and Deblurred MNIST Images
    """
    n_rows = len(imgs)
    n_cols = len(imgs[0])
    imgs = np.asarray(imgs)
    vmin = np.min(imgs)
    vmax = np.max(imgs)  # TODO maybe set it to 0,1?
    #vmin = 0
    #vmax = 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 1 + 4 * n_rows))
    if n_rows == 1:
        ax = ax[None]
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.90, top=0.95, wspace=None, hspace=None)
    for i in range(n_rows):
        if y_names is not None:
            ax[i, 0].set_ylabel(y_names[i])
        row_vmin = np.min(imgs[i])
        row_vmax = np.max(imgs[i])

        for j, x in enumerate(imgs[i]):
            im = ax[i, j].imshow(x, interpolation="none", aspect="auto", vmin=row_vmin, vmax=row_vmax)
            if names is not None and i == 0:
                ax[i, j].set_title(names[j])

            # Create shared colorbar for the row
            cax = fig.add_axes([ax[i, -1].get_position().x1 + 0.01,  # Rightmost image
                                ax[i, 0].get_position().y0,  # Align with row
                                0.02,  # Width
                                ax[i, 0].get_position().height])  # Span row height
            fig.colorbar(im, cax=cax, orientation='vertical')
    #fig.colorbar(im, ax=ax.ravel().tolist(), location="bottom")
    for ax_ in ax.flatten():
        plt.setp(ax_.get_xticklabels(), visible=False)
        plt.setp(ax_.get_yticklabels(), visible=False)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close("all")



def plot_subfigure_imgs(
    x,
    figsize=(16, 16),
    plot_title="",
    y_labels=(),
    x_labels=(),
    close=True,
    show_ticks=False,
    color_bar=False,
    show_image=True,
    save_name="",
    log_scale=False,
    vmin=None,
    vmax=None,
    folder_name="",
):
    if type(x) == list:
        h, w = len(x), len(x[0])
    else:
        h, w = x.shape[0], x.shape[1]
    fig, ax = plt.subplots(h, w, figsize=figsize)
    if log_scale:
        norm = LogNorm()
    else:
        norm = None
    if h == 1:
        ax = np.expand_dims(ax, 0)
    print(f"a plot of height {h} and width {w}", ax.shape)
    cmap = "viridis"  # 'gray'

    # Accept V_min and V_max per image
    if isinstance(vmin, list):
        # We assume vmin per row
        vmin = np.array([vmin] * w).T
    elif not isinstance(vmin, np.ndarray):
        # We assume vmin per plot
        vmin = np.array([vmin] * h * w).reshape(h, w)

    if isinstance(vmax, list):
        vmax = np.array([vmax] * w).T
    elif not isinstance(vmax, np.ndarray):
        vmax = np.array([vmax] * h * w).reshape(h, w)

    if plot_title:
        fig.suptitle(plot_title)
    if show_image:
        for i in range(h):
            if y_labels:
                ax[i, 0].set_ylabel(y_labels[i])
            for j in range(w):
                if x_labels and i == 0:
                    ax[i, j].set_title(x_labels[j])

                im = ax[i, j].imshow(
                    x[i][j], cmap=cmap, norm=norm, vmin=vmin[i, j], vmax=vmax[i, j], interpolation="none"
                )
                if color_bar:
                    divider = make_axes_locatable(ax[i, j])
                    # cax = divider.append_axes("right", size="5%", pad=0.05)
                    cax = divider.append_axes("bottom", size="5%", pad=0.05)
                    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
                    # locator, formatter = cb._get_ticker_locator_formatter()
                    # locator.set_params(nbins=3)
                    # cb.update_ticks()
                    """
                    cbar = plt.colorbar(im, ax=ax[i, j], orientation='horizontal')
                    for t in cbar.ax.get_yticklabels():
                        t.set_fontsize(10)"""
    else:

        handles, labels = ax[-1].get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center')
        fig.legend(
            handles,  # The line objects
            labels=labels,  # The labels for each line
            loc="center right",  # Position of legend
        )

    fig.subplots_adjust(right=0.70, top=0.88, left=0.05)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #plt.subplots_adjust(wspace=0, hspace=0)
    if not show_ticks:
        for ax_ in ax.flatten():
            # ax_.tick_params(axis=u'both', which=u'both',length=0)
            ax_.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=True)

            plt.setp(ax_.get_xticklabels(), visible=False)
            plt.setp(ax_.get_yticklabels(), visible=False)

            if False:
                # Create a Rectangle patch
                rect = patches.Rectangle((10, 30), 64, 128, linewidth=1, edgecolor="r", facecolor="none", fill=False)
                rect2 = patches.Rectangle((30, 120), 64, 128, linewidth=1, edgecolor="b", facecolor="none", fill=False)

                # Add the patch to the Axes
                ax_.add_patch(rect)
                ax_.add_patch(rect2)

    fig.show()
    if save_name:
        if not os.path.exists(f"plots{folder_name}"):
            os.makedirs(f"plots{folder_name}")
        # fig.savefig(
        #    f"plots{folder_name}/SF_{save_name}_{plot_title}.png", dpi=400,  bbox_inches='tight')
        fig.savefig(f"plots{folder_name}/SF_{save_name}_{plot_title}.pdf", dpi=500, bbox_inches="tight")


    # plt.close("all")


def plot_3d_coords(coords, values=None, colorbar=False, plot_brain=False, only_active=False):
    # Generate a figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    if plot_brain:
        try:
            from nilearn import datasets
            from nilearn import surface
            # Load a 3D brain model from nilearn
            fsaverage = datasets.fetch_surf_fsaverage()
            mesh_r = surface.load_surf_mesh(fsaverage['pial_right'])
            mesh = surface.load_surf_mesh(fsaverage['pial_left'])

            mesh_rscaled = mesh_r[0][:, [1, 0, 2]]
            mesh_scaled = mesh[0][:, [1, 0, 2]]

            mesh_scaled, mesh_rscaled = align_mesh(coords, mesh_rscaled, mesh_scaled)

            # Plot the brain as a transparent surface
            ax.plot_trisurf(mesh_rscaled[:, 0], mesh_rscaled[:, 1], mesh_rscaled[:, 2], triangles=mesh_r[1],
                            color='gray',
                            alpha=0.1)
            ax.plot_trisurf(mesh_scaled[:, 0], mesh_scaled[:, 1], mesh_scaled[:, 2], triangles=mesh[1], color='gray',
                            alpha=0.1)

        except:
            print("Could not import nilearn")

    cmap = "binary" #"viridis"
    if only_active and values is not None:
        non_zero = values.abs() > 0.4
        # Scatter plot with point size based on values

        im = ax.scatter(coords[non_zero, 0], coords[non_zero, 1], coords[non_zero, 2], c=values[non_zero], s=50, alpha=.8, cmap=cmap)
        im = ax.scatter(coords[~non_zero, 0], coords[~non_zero, 1], coords[~non_zero, 2], c=values[~non_zero]*0, s=30,
                        alpha=0.1, cmap=cmap)

    else:

        im = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values, s=50, alpha=0.6, cmap=cmap)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set interactive mode
    plt.ion()

    if colorbar:
        im.set_clim(0, values.max())
        fig.colorbar(im, ax=ax, location="bottom")


    # Show the plot
    plt.show()


def align_mesh(coords, mesh_l, mesh_r):
    # Find the minimum and maximum values along each axis for both coordinates and mesh
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)

    min_mesh = np.min(np.concatenate([mesh_l, mesh_r], axis=0), axis=0)
    max_mesh = np.max(np.concatenate([mesh_l, mesh_r], axis=0), axis=0)

    # Calculate scaling factors for each axis
    scale_factors = (max_coords - min_coords) / (max_mesh - min_mesh)

    # Apply scaling to the mesh
    scaled_mesh_l = (mesh_l - min_mesh) * scale_factors[None, :] + min_coords
    scaled_mesh_r = (mesh_r - min_mesh) * scale_factors[None, :] + min_coords

    return scaled_mesh_l, scaled_mesh_r


def read_and_combine_df(models, mode="", dataset="MNEVol", time=""):
    df = None
    for m in models:

        try:
            files = os.listdir(f"../lightning_logs/{m}/")
        except:
            print("Could not find model ", m)
            continue

        # if f"Eval-{mode}-{dataset}.csv" in files:

        base_name_file = "Eval" if f"Eval{time}-{mode}-{dataset}.csv" in files else "Evaluation"
        base_name_file += time

        if f"{base_name_file}-{mode}-{dataset}.csv" in files:
            df_tmp = pd.read_csv(f"../lightning_logs/{m}/{base_name_file}-{mode}-{dataset}.csv", index_col=0)

            if "NSensors" in mode:
                df_tmp["SensorSeed"] = [int(re.search(r'seed(\d+)', x).group(1)) for x in df_tmp.index]
                df_tmp["NumSensor"] = [int(re.search(r'sensor(\d+)', x).group(1)) for x in df_tmp.index]


        else:

            print(f"Could not find model {m}, DS {dataset}")
            continue
        try:
            if mode == "timesteps":
                pattern = r".*generateTimeFreq(.*)_N.*"
                df_tmp["TimeFreq"] = [re.match(pattern, x).group(1) for x in df_tmp.index]
            pattern = r".*nf(.*)generate.*_N(.*)_S(.*)_E(.*)_flf([0-9])_ilf([0-9])"
            df_tmp["Noisefree"] = [re.match(pattern, x).group(1) for x in df_tmp.index]
            df_tmp["NoiseLevel"] = [re.match(pattern, x).group(2) for x in df_tmp.index]
            df_tmp["N_Sources"] = [re.match(pattern, x).group(3) for x in df_tmp.index]
            df_tmp["Std_Sources"] = [re.match(pattern, x).group(4) for x in df_tmp.index]
            df_tmp["LeadField"] = [re.match(pattern, x).group(5) for x in df_tmp.index]
            df_tmp["InvLeadField"] = [re.match(pattern, x).group(6) for x in df_tmp.index]

        except:
            print("String not found")

        df_tmp["Sigma"] = ["Sigma" in m] * len(df_tmp.index)
        df_tmp["SSL"] = ["SSL" in m] * len(df_tmp.index)
        df_tmp["LF"] = [x.replace(m, "").replace("Pseudo", "") for x in df_tmp.index]
        if 'Noise' not in df_tmp.columns:
            numbers = re.findall(r"\d+\.\d+|\d+", m)
            if len(numbers) == 2:
                # If we have Depth in it, it is model size and depth
                if "Depth" in m:
                    df_tmp["HiddenDim"] = [float(numbers[0])] * len(df_tmp.index)
                    df_tmp["Depth"] = [float(numbers[1])] * len(df_tmp.index)
                else:
                    # we Have two numbers and asume noise and blur
                    df_tmp["Noise"] = [float(numbers[0])] * len(df_tmp.index)
                    df_tmp["Blur"] = [float(numbers[1])] * len(df_tmp.index)
            else:
                df_tmp["Noise"] = ["Noise" in m] * len(df_tmp.index)
        df_tmp["Measure"] = ["--include_measurements" in m] * len(df_tmp.index)
        # df_tmp["ModelName"] = [m.replace("Sup","").replace("EEG","").replace("32",""),"Pseudo"]
        # df_tmp["ModelName"] = [x.replace("Sup","").replace("EEG","").replace("32","").replace("MultiStrongForward","") for x in df_tmp.index]
        if m[:-1].endswith("S") or m[:-1].endswith("Seed"):
            # Hacky way to select seed
            df_tmp["Seed"] = m[-1]
        else:
            df_tmp["Seed"] = 42  # Default Seed
        df_tmp["ModelName"] = [("Pseudo" if x.startswith("Pseudo") else m.replace("Sup", "").replace("EEG", "").replace(
            "32", "").replace("MultiStrongForward", "").rstrip("S1234567890")) for x in df_tmp.index]
        df_tmp["FullName"] = ["Pseudo" if x.startswith("Pseudo") else m for x in df_tmp.index]
        df_tmp["Tuned"] = [x.startswith("Tuned") for x in df_tmp.index]
        df_tmp["Dataset"] = dataset
        if df is None:
            df = df_tmp
        else:
            try:
                df_tmp = df_tmp.drop(index="PseudoInverse")
            except:
                pass
            df = pd.concat([df, df_tmp])

    return df


def combine_multiple_datasets(models, mode="", datasets=["MNEVol", ], time=""):
    df = read_and_combine_df(models, mode, datasets[0], time)

    for ds in datasets[1:]:
        df_tmp = read_and_combine_df(models, mode, ds, time)
        df = pd.concat([df, df_tmp])
    return df


def clean_up(df, name_map_updated=None):
    # df = df[df["Std_Sources"]!="0.3-0.3"]

    name_map = {
        x: x.replace("Sup", "").replace("EEG", "").replace("32", "").replace("MultiStrongForward", "").replace("Linear",
                                                                                                               "FCN").replace(
            "Seed", "").rstrip("S1234567890").replace("MNEVol", "").replace("Depth", "").replace("Unet", "3D-PIUNet")
        for x in df["FullName"]}
    if name_map_updated is not None:
        name_map = { **name_map,**name_map_updated}
    df["Model"] = [name_map[x] for x in df["FullName"]]

    df["SNR"] = [int(x.split("-")[0]) for x in df["NoiseLevel"]]
    df["Source Extent"] = [float(x.split("-")[1]) * 200 for x in df["Std_Sources"]]
    df["Number of Sources"] = [int(x.split("-")[1]) for x in df["N_Sources"]]
    df["Source Range"] = [x for x in df["N_Sources"]]
    df["Weighted Cosine"] = [1 - x for x in df["WeightedAngularError"]]
    if "NormalizedEMD" in df.columns:
        df["Normalized EMD"] = [x for x in df["NormalizedEMD"]]
    else:
        df["Normalized EMD"] = [0 for x in df["WeightedAngularError"]]
    if "TimeFreq" in df.columns:
        df["TimeFreq"] = [int(x) for x in df["TimeFreq"]]

    df["Normalized MSE"] = [x for x in df["NormalizedMSE"]]
    # df["WeightedCosine"] = [x for x in df["WeightedAngularError"]]

    return df