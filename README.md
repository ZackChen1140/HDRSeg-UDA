# HDRSeg-UDA: Unsupervised Domain Adaptation for Semantic Segmentation of High Dynamic Range Images under Adverse Conditions

The training framework leverages unsupervised domain adaptation to enable robust semantic segmentation across diverse weather conditions. Unlike standard images, High Dynamic Range (HDR) images contain significantly higher bit depth, allowing for richer visual information. By extracting the enhanced feature representations in HDR images, the model achieves improved segmentation performance, especially under extreme weather scenarios.

![overview of training framework](figure/Dual_Path_SegFormer_UDA_Training_structure(dark).png)

## Environment

You can build a docker image by *dockerfile*,
```
docker build -t hdrseg .
```
and just run *run_docker.sh* to create a container and access the terminal.
```
chmod +x run_docker.sh # Only required for the first time
./run_docker.sh
```
Or
```
docker run -it \
    --name="hdrseg" \
    --gpus=all \
    --shm-size=32g \
    --volume=".:/home/app/HDRSeg-UDA" \
    hdrseg /bin/bash
```

---

If you prefer to create a virtual environment in local, you can install [PyTorch](https://pytorch.org/) first.

Then install the requirements with

```
pip install -r requirements.txt
```

## Execution

Before training, category statistics file for the clear training data should be generated.
```
python -m tools.count_categories <path/to/your/categories.csv> <path/to/your/training_images> <path/to/your/training_labels> <path/to/your/output_file>
```

For example,
```
python -m tools.count_categories data/csv/rlmd.csv data/rlmd_ac/clear/train/images data/rlmd_ac/clear/train/labels data/rcs_files/rlmd.json
```

Or you just want to see the proportion of each category,
```
python -m tools.count_categories data/csv/rlmd.csv data/rlmd_ac/clear/train/images data/rlmd_ac/clear/train/labels
```

---

To starting training, please choose a configuration(you can also tune hyperparameters by yourself).
```
python -m tools.train_uda <path/to/your/config>
```
For example, 
```
python -m tools.train configs/train_uda_rlmdac_dual_clear_to_mix.json
```

---

To resume training from an interupted experiment:
```
python -m tools.train <path/to/your/config> <path/to/your/logs/experiment/checkpoint>
```
For example, 
```
python -m tools.train configs/train_uda_rlmdac_dual_clear_to_mix.json logs/train_uda_rlmdac_dual_clear_to_mix/latest_checkpoint.pth
```

---

To check the training progress, this framework support tensorboard.
```
tensorboard --logdir <path/to/your/log/experiment>
```
For example,
```
tensorboard --logdir logs/train_uda_rlmdac_dual_clear_to_mix_202504180/
```

---

To evaluate the trained model, please choose a evaluation configuration, and the checkpoint:
```
python -m tools.evaluation <path/to/your/config> <path/to/your/logs/experiment/checkpoint>
```
For example,
```
python -m tools.evaluation configs/evaluate_rlmdac_dual_clear_night_rainy.json logs/train_uda_rlmdac_dual_clear_to_mix/latest_checkpoint.pth
```

---

by using custom dataset, please ensure that the labels are stored in P-mode (palette mode).

First, you should create a category csv file for custom dataset, you can follow the format in the [given ones](data/csv/rlmd.csv).
```
python -m tools.convert_to_p_mode <path/to/your/category/csv> <path/to/your/labels> <path/to/your/output>
```