# TA-HPC-Scripts

This repository contains codes used for training models of [Deeplanding](https://github.com/perfect-less/deeplanding) autoland system. Make sure you've already trimmed your data (as written [here](https://github.com/perfect-less/deeplanding)) before you train your models with it using this codes.

Make sure you're working in `TA-HPC-Scripts` folder before proceeding.
```bash
$ cd TA-HPC-Scripts/
```

## Setting up your data

1. If it didn't already exist, create `Data/Processed/Selected/` directory inside `TA-HPC-Scripts/`. Your directory structure should look  like this:
```bash
$ deeplanding
$ ├── ...
$ └── TA-HPC-Scripts
$     └── Data
$         └── Processed
$             └── Selected
$ ├── ...
$ ...
```
2. Now, copy all the trimmed data into `Selected` folder.
3. If you want to train longitudinal models put all files inside `deeplanding/xpcclient/Trimmed.lon` into `TA-HPC-Scripts/Data/Processed/Selected/`. And if you want to train aileron model, replace it with files from `deeplanding/xpcclient/Trimmed.ail`.


## Train your models
First, we will train the longitudinal models, it contains elevator model, throttle model, selected airspeed model, and flap model.
1. Make sure you're working in `TA-HPC-Scripts` folder.
```bash
$ cd TA-HPC-Scripts/
```
2. Train the models one-by-one. Make sure to remember the order in which you train the models, because we need to rename the folder containing our models later.
```bash
## Bellow are the command to train each model

# 1. Elevator model
$ ./hpcs --mid e_simp_1
$ ./hpcs --from trainready --until post

# 2. Throttle model
$ ./hpcs --mid t_simp_1
$ ./hpcs --from trainready --until post

# 3. Selected airspeed model
$ ./hpcs --mid s_simp_1
$ ./hpcs --from trainready --until post

# 4. Flap model
$ ./hpcs --mid f_simp_1
$ ./hpcs --from trainready --until post
```
3. Rename each respective models' folder into into `Elevator`, `Throttle`, `SelectedAirspeed`, and `Flap`. And then copy those folder into `deeplanding/xpcclient/Models/`.

Now that we finished training longitudinal models, next we will train the aileron model.

4. Rename `TA-HPC-Scripts/Data/Processed/` into `TA-HPC-Scripts/Data/Processed.lon` (so we can accessed the data inside it later).
5. Copy all files from `deeplanding/xpcclient/Trimmed.ail` to `TA-HPC-Scripts/Data/Processed/Selected/`. 
6. Train the aileron model
```bash
# Train aileron model with the following command
$ ./hpcs --mid a_simp_1
$ ./hpcs --from trainready --until post
```
7. Rename the model's folder to `Aileron` and then copy that folder to `deeplanding/xpcclient/Models/`.

## Check models' performance on training data
You can also test the models's prediction on training data by checking the [compare_model.ipynb](compare_model.ipynb) notebook.