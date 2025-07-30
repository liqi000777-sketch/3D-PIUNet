

python supervised_model.py --accelerator cuda  --network_architecture unet --experiment_name SupMNEVolUnet32Seed1 --seed 1
python supervised_model.py --accelerator cuda  --network_architecture linear --network_size 1024 --experiment_name SupMNEVolLinear1024Seed1 --seed 1
python supervised_model.py --accelerator cuda  --network_architecture ConvDip --experiment_name SupMNEVolConvDipSeed1 --seed 1
python supervised_model.py --accelerator cuda  --max_epochs 1 --network_architecture SupeLORETA --experiment_name SupeLORETA
python supervised_model.py --accelerator cuda  --max_epochs 1 --network_architecture SupLassoPosZero --experiment_name SupLassoPosZero

models=(
        "SupMNEVolUnet32Seed1"
        "SupMNEVolLinear1024Seed1"
        "SupMNEVolConvDipSeed1"
        "SupeLORETA"
        "SupLassoPosZero"
)
config=("snr" "source_std" "Nsources")
data_sets=("MNEVolSUBJECTOAS004")
for model in "${models[@]}"; do
  for conf in "${config[@]}"; do
    for ds in "${data_sets[@]}"; do
      echo "$model and $conf and $ds"
      python evaluation_script.py --cuda --batch_size 32 "$model" --task_config "$conf" --max_length 200 --base_data "$ds" --compute_emd
    done
  done
done
