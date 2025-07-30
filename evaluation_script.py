import numpy as np
import torch
import os
import re

from supervised_model import SupervisedEEGModel
import pandas as pd
from utils import metrics
import argparse
import tqdm
from utils.data_utils import load_compressed_pickle
from utils.things_eeg import get_post_stimulus_data
from conditioning.EEG_forward_model import EEGforward
from conditioning.utils import mkfilt_eloreta_v2
from distributions.pointdistribution import PointDistribution
import functools

def load_model(model_name, cuda=False):

    try:
        checkpoints = os.listdir(f"lightning_logs/{model_name}/checkpoints/")
        path_to_checkpoint = f"lightning_logs/{model_name}/checkpoints/"
    except FileNotFoundError:
        print(f"No model {model_name} found!")
        look_for_checkpoint = os.listdir(f"lightning_logs/{model_name}/EEG_Project/")
        for ck in look_for_checkpoint:
            if "checkpoints" in os.listdir(f"lightning_logs/{model_name}/EEG_Project/{ck}"):
                checkpoints = os.listdir(f"lightning_logs/{model_name}/EEG_Project/{ck}/checkpoints/")
                path_to_checkpoint = f"lightning_logs/{model_name}/EEG_Project/{ck}/checkpoints/"
    if "final_model.ckpt" in checkpoints:
        newest_ckpt = "final_model.ckpt"
    else:
        newest_ckpt = max(checkpoints, key=lambda x: int(x.split("step=")[-1].replace(".ckpt", "")))

    if torch.cuda.is_available() and cuda:
        device = "cuda:0"
    else:
        device = "cpu"
    model = SupervisedEEGModel.load_from_checkpoint(
        f"{path_to_checkpoint}{newest_ckpt}", map_location=device, strict=False
    )

    model = model.to(device)
    return model

def eval_sn_set(model, model_name, totaloss_df = None, test_mode="snr", base_data="eegHBN", max_length=200, time_steps=1, compute_emd=True):
    ########## EXperiments with HBN model!  ##############
    correlated_noise = [0]
    if test_mode == "snr":
        sn_set = ["0-0", "1-1", "3-3", "5-5", "10-10", "15-15", "20-20", "30-30", "40-40"] # ["1-1", "10-10", "20-20"]
        ns_set = ["1-1"]#, "1-10"]#, "5-5"] #["1-1", "6-6", "10-10"]
        sd_set = ["0.05-0.4"] # "0.05-0.05",["0.005-0.005"] #, "0.05-0.3"]
    elif test_mode == "source_std":
        sn_set = ["5-5"]
        ns_set = ["1-1"]
        sd_set = ["0.05-0.05", "0.1-0.1", "0.15-0.15", "0.2-0.2", "0.3-0.3", "0.4-0.4"]
    elif test_mode == "Nsources":
        sn_set = ["20-20"]
        ns_set = ["1-1", "2-2", "3-3", "5-5", "8-8", "10-10"]
        sd_set = ["0.05-0.4"]
    elif test_mode == "small":
        sn_set = ["0-0", "5-5", "20-20"]
        ns_set = ["1-1", "1-10"]
        sd_set = ["0.05-0.4"]
    elif test_mode == "corr_noise":
        sn_set = ["5-5", "20-20"]
        ns_set = ["1-1"]
        sd_set = ["0.05-0.4"]
        correlated_noise = [0.0, 0.05, 0.1, 0.2, 0.3]
    elif test_mode == "timesteps":
        sn_set = ["5-5"]
        ns_set = ["1-10"]
        sd_set = ["0.05-0.4"]
    else:
        raise ValueError("Test mode not known!")
    if base_data=="eegHBN":
        model.forward_model = EEGforward(mode=f"HBNall")
        fw_lfs = [5,6]
        inv_lfs = [4, 5]

    elif base_data == "MNEVol":
        model.forward_model = EEGforward(mode=f"MNEVolboth")
        fw_lfs = [1]
        inv_lfs = [0]
    elif "MNEVolSUBJECT" in base_data:
        model.forward_model = EEGforward(mode=base_data)
        fw_lfs = [1]
        inv_lfs = [0]

    else:
        raise ValueError(f"Data Set {base_data} not known!")


    if time_steps > 1:
        time_add = f"TimeFreq{time_steps}"
    else:
        time_add = ""

    model.set_time_step(time_steps)
    # Recompute Covariance if MNE inverse is used:
    recompute_cov = model_name in ["SupGammaMap", "SupMNEeLORETA", "SupMNE", "SupdSPM", "SupsLORETA", "Supbeamformer"]

    for forward_lf in fw_lfs:
        for inverse_lf in inv_lfs:
            lfloss_df = None
            for sn in sn_set:
                for ns in ns_set:
                    for sd in sd_set:
                        for c_n in correlated_noise:
                            if c_n:
                                model.forward_model.set_noise_pattern(0, c_n)
                                dataset = f"{base_data}generate{time_add}CorrNoise{c_n}_N{sn}_S{ns}_E{sd}"
                            else:
                                dataset = f"{base_data}generate{time_add}_N{sn}_S{ns}_E{sd}"
                            setting = dataset + f"_flf{forward_lf}_ilf{inverse_lf}"
                            model.hparams.data_set = dataset
                            for nf in [False]:
                                df_tmp = model.evaluate_model(noise_free=nf, forward_index=inverse_lf, plot_add=setting,
                                                              data_leadfield_index=forward_lf, max_length=max_length,
                                                              recompute_cov=recompute_cov, compute_emd=compute_emd)
                                                              #prediction_path=f"lightning_logs/{model_name}/Evaluation-{dataset}-L{max_length}")
                                if lfloss_df is None:
                                    lfloss_df = df_tmp.copy()
                                else:
                                    lfloss_df = pd.concat([lfloss_df, df_tmp])

            lfloss_df.to_csv(
                f"lightning_logs/{model_name}/Evaluation{time_add}-{test_mode}-{base_data}L{max_length}LF{forward_lf}ILF{inverse_lf}.csv")

            if totaloss_df is None:
                totaloss_df = lfloss_df.copy()
            else:
                totaloss_df = pd.concat([totaloss_df, lfloss_df])

    totaloss_df.to_csv(
        f"lightning_logs/{model_name}/Evaluation{time_add}-{test_mode}-{base_data}L{max_length}.csv")
    return totaloss_df

def number_sensors_experiment(model, model_name, base_data="MNEVol", max_length=200):
    assert base_data == "MNEVol", f"Data Set {base_data} not known!"
    model.forward_model = EEGforward(mode="MNEVolboth")
    fw_lf = 1
    inv_lf = 0
    # We select a subset of sensors and evaluate the performance
    #sensor_list = [4,8,16,32,model.forward_model.Nsens]
    sensor_list = [model.forward_model.Nsens-i for i in range(1,12,2)]
    seed_list = [1,2,3,4,5]
    totaloss_df = None
    for snr, ns in [(20,1),(20,3),(10,1),(10,3)]:
        dataset = f"{base_data}generate_N{snr}-{snr}_S{ns}-{ns}_E0.05-0.4"
        setting = dataset + f"_flf{fw_lf}_ilf{inv_lf}"
        model.hparams.data_set = dataset
        # Generate the Dataset once with all sources active
        model.val_dataloader(forward_index=fw_lf, max_length=max_length)
        for ns in sensor_list:
            for seed in seed_list:
                model.forward_model = EEGforward(mode="MNEVolboth")
                sensor_subset = torch.randperm(model.forward_model.Nsens, generator=torch.Generator().manual_seed(seed))[:(model.forward_model.Nsens-ns)]

                model.forward_model.select_sensors_subset(sensor_subset)
                df_tmp = model.evaluate_model(noise_free=False, forward_index=inv_lf, plot_add=setting+f"seed{seed}"+f"sensor{ns}",
                                              data_leadfield_index=fw_lf, max_length=max_length, sensor_subset=sensor_subset, compute_emd=True)
                if totaloss_df is None:
                    totaloss_df = df_tmp.copy()
                else:
                    totaloss_df = pd.concat([totaloss_df, df_tmp])

        totaloss_df.to_csv(
            f"lightning_logs/{model_name}/Eval-NSensors-{dataset}L{max_length}.csv")



def eval_Esinet(model, model_name,totaloss_df = None):

    for ds in ["esinetoneD","ico3esinetoneD"]:
        # Try Esinet with fsaverage forward:
        model.hparams.data_set = ds
        model.forward_model = EEGforward(mode=f"MNE-{ds}")
        if "transformer" in model.hparams.network_architecture:
            model.eeg_net.set_source_embedding(torch.tensor(model.forward_model.scs_loc).float())
            model.eeg_net.set_meas_embedding(torch.tensor(model.forward_model.sens_loc).float())

        for nf in [False, True]:
            df_tmp = model.evaluate_model(noise_free=nf, forward_index=0,
                                                       plot_add=f"{ds}{nf}")

            if totaloss_df is None:
                totaloss_df = df_tmp.copy()
            else:
                totaloss_df = pd.concat([totaloss_df, df_tmp])

    totaloss_df.to_csv(
        f"lightning_logs/{model_name}/Evaluation-results-withMNE.csv")
    return totaloss_df

def eval_real_data(model, model_name, averaged=False, max_length=200, ses=1, time_steps=1):
    # Set the forward model to correct forward
    model.forward_model = EEGforward(mode="MNEVolthingseeg").to(model.device)

    # As we do not have a ground truth, we can only investigate predictions and Pseudo Inv
    data = get_post_stimulus_data("test", averaged=averaged, L=max_length, ses=ses)
    # Data has the shape: Images x Time x Channels
    # We have 600 time steps!
    if time_steps>1:
        model.set_time_step(600)
    else:
        model.set_time_step(1)
    results = []
    pseudo = []
    for x in tqdm.tqdm(data):
        with torch.no_grad():
            sensors = torch.tensor(x).float().to(model.device)


            pseudo_out = model.forward_model.pseudo_inv_specific(sensors, forward_index=0)
            cond_variable = model.get_cond_variable({"sensors":sensors}, forward_index=0)
            out = model.forward(sensors, cond_variable, forward_index=0)

        if isinstance(out, torch.distributions.Distribution):
            out = out.mean
        # Tuning predictions disrupt performance!
        #out = model.tune_prediction_pytorch(sensors, out, forward_index=0)
        results.append(out.detach().cpu().numpy())
        pseudo.append(pseudo_out.detach().cpu().numpy())
    results = np.array(results)
    pseudo = np.array(pseudo)
    np.save(f"lightning_logs/{model_name}/Things_testdata{'_averaged' if averaged else ''}_ses{ses}_resultsL{max_length}.npy", results)
    np.save(f"lightning_logs/{model_name}/Things_testdata{'_averaged' if averaged else ''}_ses{ses}_pseudoL{max_length}.npy", pseudo)


def run_time_experiment():
    models_to_show = [
                      "SupMNEVolLinear512Depth3", "SupMNEVolLinear1024Depth3", "SupMNEVolLinear1024Depth4",
                      "SupMNELinear4096Depth3", "SupMNEVolLinear16384Depth3", "SupMNEVolConvDipSeed1",
                      "SupMNEVolUnet8Depth3", "SupMNEVolUnet32Depth3", "SupMNEVolUnet32Depth4", "SupMNEVolUnet64Depth3",
                      "eLORETA", "SupLasso", "SupeLORETA", "Lasso_SciPy"
                      ]
    models_to_show = ["SupMNEVolUnet32Seed1",
        "SupMNEVolLinear1024Seed1",
        "SupMNEVolConvDipSeed1",
        "SupeLORETA",
        "SupLassoPosZero",
                      ]

    models_to_show = [
                        "SupeLORETA",
                        "SupMNEVolLinear1024Seed1",
                      ]
    dataset = "MNEVolSUBJECTOAS004generate_N20-20_S1-1_E0.05-0.4"
    setting = dataset + f"_flf{1}_ilf{0}"
    import time
    times_per_model = {}
    losses_per_model = {}
    for model_name in models_to_show:
        if model_name == "eLORETA":
            model = load_model("SupeLORETA", True)
        elif model_name == "Lasso_SciPy":
            model = load_model("SupLasso", True)
        else:
            model = load_model(model_name, True)

        model.forward_model = EEGforward(mode="MNEVolSUBJECTOAS004", device=model.device)
        model.hparams.batch_size = 500
        model.hparams.data_set = dataset
        val_loader = model.val_dataloader(forward_index=1, max_length=None)
        times_per_model[model_name] = []

        for i in range(50):
            used_time = 0
            tar, pred, sens = [], [], []
            for batch in val_loader:
                with torch.no_grad():
                    if batch["sources"].device != model.device:
                        # a little bit ugly, as source centers is a list of tensors!
                        batch = {k: v.to(model.device) if k != "source_centers" else [c.to(model.device) for c in v] for k, v in batch.items()}
                    cond_variable = model.get_cond_variable(batch, forward_index=0)
                    start_time = time.time()
                    out_dist = model.forward(batch["sensors"], cond_variable, forward_index=0)
                    if isinstance(out_dist, torch.distributions.Distribution):
                        out_dist = out_dist.mean
                    if model.forward_model.padding:
                        target = model.forward_model.remove_padding(target)
                        out_dist = model.forward_model.remove_padding(out_dist)
                        pseudo_out = model.forward_model.remove_padding(pseudo_out)
                    used_time += time.time() - start_time
                    if i==0:
                        tar.append(batch["sources"].detach().cpu())
                        pred.append(out_dist.detach().cpu())
                        sens.append(batch["sensors"].detach().cpu())

            times_per_model[model_name].append(used_time)
            if i ==0:
                tar = torch.cat(tar, dim=0)
                pred = torch.cat(pred, dim=0)
                sens = torch.cat(sens, dim=0)
                losses = metrics.compute_metrics(tar, pred, forward_model=model.forward_model, forward_index=1,
                                                 measurements=sens, compute_emd=False) # TODO compute emd
                losses_per_model[model_name] = {k: torch.mean(v).numpy() for k,v in losses.items()}
        print(f"Model {model_name} took {np.mean(times_per_model[model_name])} seconds on average!")
        num_params = sum(p.numel() for p in model.eeg_net.parameters())
        print("The model had ", num_params, " parameters!")
        print("Losses: ", losses)
    # Save the times as df
    times_df = pd.DataFrame(times_per_model)
    times_df.to_csv(f"TimeComparisonSingleMinorRevisionCompiled-{dataset}.csv")
    # Save the losses as df
    losses_df = pd.DataFrame(losses_per_model)
    losses_df.to_csv(f"LossesMinorRevisionCompiled-{dataset}.csv")


def tune_hyperparameter(model_name, cuda, max_length=2000, dataset="MNEVolgenerate_N0-30_S1-1_E0.05-0.4"):
    assert model_name in ["SupLasso", "SupLassoZero", "SupLassoPosZero", "SupLassoPos", "SupGammaMap", "SupeLORETA","SupLasso_SciPy", "SupChampagne"], "Hyper Parameter Tune not available for this model!"
    model = load_model(model_name, cuda)
    model.hparams.data_set = dataset
    forward_index = 0
    model.forward_model = EEGforward(mode="MNEVolboth")
    model.hparams.batch_size = 64
    val_loader = model.val_dataloader(forward_index=forward_index, max_length=max_length)
    # We recompute the covariance matrix for the evaluation on the full dataset
    if isinstance(val_loader.dataset, torch.utils.data.Subset):
        # For Subsets, we need to access the original data first:
        model.forward_model.set_mne_obj(torch.tensor(val_loader.dataset.dataset.data["sensors"]))
    else:
        model.forward_model.set_mne_obj(torch.tensor(val_loader.dataset.data["sensors"]))


    #hyper_param_set = np.linspace(0.0,400,41) if model_name in ["SupLasso", "SupLassoPos","SupLassoZero", "SupLassoPosZero","SupLasso_SciPy"] else np.linspace(0, 1, 41)
    hyper_param_set = np.logspace(-1, 6, 50) if model_name in ["SupLasso", "SupLassoPos", "SupLassoZero",
                                                                  "SupLassoPosZero", "SupLasso_SciPy"] else np.linspace(
        0, 1, 41)

    losses = {}
    loss = torch.nn.L1Loss(reduction="sum")
    for hyperparam in hyper_param_set:
        if "eLORETA" in model_name:
            model.forward_model.forward_matrices[forward_index]["P"] = torch.tensor(mkfilt_eloreta_v2(
                model.forward_model.forward_matrices[forward_index]["L"], regu=hyperparam)).float().reshape(model.forward_model.Nsens,model.forward_model.Nsources,-1)
        else:
            model.eeg_net.set_hyperparam(hyperparam)

        total_loss = 0
        EMD_loss = 0
        NMNSE_loss = 0
        NEMD_loss = 0
        for x in val_loader:
            sensors = x["sensors"].to(model.device)
            target = x["sources"].to(model.device)
            out = model.forward(sensors, forward_index=forward_index)
            if isinstance(out, torch.distributions.Distribution):
                out = out.mean
            #out = torch.nan_to_num(out,nan=1e-8)
            if torch.isnan(out).sum() > 0:
                print(f"Found NaN in output for hyperparam {hyperparam}!")
                continue
            total_loss += loss(out, target).item() / model.forward_model.Nsources
            NMNSE_loss += metrics.normalized_mean_squared_error(target, out).detach().cpu().sum().numpy()
            emd, normalized_emd = metrics.compute_emd_and_sinkhorn(target, out, model.forward_model)
            EMD_loss += np.sum(emd)
            NEMD_loss += np.sum(normalized_emd)
        losses[hyperparam] =(total_loss, NMNSE_loss, EMD_loss, NEMD_loss)
    losses = pd.DataFrame(losses, index=[model_name+"MAE",model_name+"NMSE", model_name+"EMD", model_name+"NEMD_loss"])
    losses.to_csv(f"lightning_logs/{model_name}/SingleHyperParameterTuningL{dataset}{max_length}.csv")

def load_and_evaluate(model_name, noise_free=False, batch_size=32, cuda=False, task_config="", plot_add="", max_length=200, time_steps=1, base_data="MNEVol", compute_emd=False):
    model = load_model(model_name, cuda)
    device = model.device
    model.hparams.batch_size = batch_size
    totaloss_df = None

    # Save the number of parameter in an overview file.
    num_params = sum(p.numel() for p in model.eeg_net.parameters())
    with open("overview.txt", "a+") as f:
        f.write(f"{model_name}  ParCount: {num_params} \n")

    if "realData" in task_config:
        if "ses" in task_config:
            ses = int(re.findall(r'\d+', task_config)[0])
        else:
            ses = 1
        eval_real_data(model, model_name, averaged="averaged" in task_config, max_length=max_length, ses=ses, time_steps=time_steps)
    elif "number_sensors" in task_config:
        number_sensors_experiment(model, model_name, max_length=max_length)
    elif task_config:
        if task_config == "timesteps":
            time_steps_var = [2, 10, 50, 100, 200, 500]
            for time_steps in time_steps_var:
                totaloss_df = eval_sn_set(model, model_name, test_mode=task_config, base_data=base_data,
                                      max_length=max_length, time_steps=time_steps, compute_emd=compute_emd)
        else:
            totaloss_df = eval_sn_set(model, model_name, test_mode=task_config, base_data=base_data, max_length=max_length, time_steps=time_steps, compute_emd=compute_emd)

    #totaloss_df = eval_Corr_noise(model, model_name)
    #totaloss_df = eval_Esinet(model, model_name)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--noise_free", default=False, action="store_true")
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--task_config", default="", type=str)
    parser.add_argument("--max_length", default=200, type=int, help="The number of samples to evaluate on")
    parser.add_argument("--time_steps", default=1, type=int)
    parser.add_argument("--base_data", default="MNEVol", type=str)
    parser.add_argument("--compute_emd", default=False, action="store_true")
    parser.add_argument("--dataset_add", default="generate_N0-30_S1-1_E0.05-0.4", type=str)
    args = parser.parse_args()
    if "hyperparam_tune" in args.task_config:
        tune_hyperparameter(args.model_name, args.cuda, args.max_length, dataset= args.base_data+args.dataset_add)
    elif "runtime" in args.task_config:
        run_time_experiment()
    else:
        load_and_evaluate(
            model_name=args.model_name,
            noise_free=args.noise_free,
            batch_size=args.batch_size,
            cuda=args.cuda,
            task_config=args.task_config,
            max_length=args.max_length,
            time_steps=args.time_steps,
            base_data=args.base_data,
            compute_emd=args.compute_emd,
            # plot_add=plot_add,
        )
