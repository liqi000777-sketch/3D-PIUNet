# Enhancing Brain Source Reconstruction by Initializing 3D Neural Networks with Physical Inverse Solutions (3D-PIUNet)

This repository contains the code for the 3D Pseudo Inverse U-Net (3D-PIUNet) model, a deep learning model for EEG source localization that effectively integrates the strengths of traditional and deep learning techniques.
3D-PIUNet starts from an initial physics-informed estimate by using the pseudo inverse to map from measurements to source space. Secondly, by viewing the brain as a 3D volume, it uses 3D convolutional U-Net to capturing spatial dependencies and refining the solution according to a learned data prior. 
Training the model relies on simulated pseudo-realistic brain source data, covering different source distributions. Trained on this data, our model significantly improves spatial accuracy, demonstrating superior performance over both traditional and end-to-end data-driven methods.

## Installation

It is necessary to install the dependencies before running the code. To do so, run the following command:
```bash
pip install -r requirements.txt
```

## Usage

To train the 3D-PIUNet model, run the following command:
```bash
python supervised_model.py --accelerator cuda --data_set MNEVolgenerate_N0-30_S1-20_E0.005-0.04 --experiment_name TestUNet
```
It automatically generates a dataset with SNR 0-30dB, 1-20 sources, and source_std 0.005-0.04. The trained model is saved in the `lightning_logs` folder.

To evaluate the trained model on different settings (snr, Nsources, source_std), run the following command:
```bash
python evaluation_script.py TestUNet --cuda --task_config snr
```

To train and evaluate all models present in the paper (with a single seed in contrast to the 5 seeds used in the paper), run the following command:
```bash
./run_all_models.sh 
```
Training 3D-PIUNet on a single seed takes around **5 hours** on a single GPU, while the other models are faster.

However, computing the EMD on the evaluation samples is slow. We therefore limited the number of samples to evaluate on to 200 in this script (via the `--max_length 200` parameter), while the paper used 2000.

After training and evaluation is done, the main plots from the paper can be generated using the Jupiter notebook `paper_plots.ipynb`.


## Citation

If you use this code in your research, please cite the following paper:

```
@article{morik2024enhancing
  title={Enhancing Brain Source Reconstruction through Pseudo-Inverse Initialization and 3D Neural Networks},
  author={Morik, Marco and Hashemi, Ali and Müller, Klaus-Robert and Haufe, Stefan and Nakajima,Shinichi},
    journal={arXiv preprint},
    year={2024}
}
```

