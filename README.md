# Generate Image Dataset and Train Model

[中文](README_CN.md)

Our experimental environment is Ubuntu 20.04 LTS operating system and the hardware configuration is NVIDIA RTX 3090 GPU for accelerating model training.

## Setting Up Conda Environment

### Modifying `TreeCV.yml` File

To set up the Conda environment, make the following modifications to the `TreeCV.yml` file:

Update the `prefix` field with the desired installation path. For example:

```yaml
prefix: your_installation_path
```

### Creating Conda Environment
After modifying the TreeCV.yml file, use the following command to create the new environment:
```shell
conda env create -f TreeCV.yml
```
This command will create and activate the Conda environment with all the specified dependencies.

## Generate Image Dataset

1. First, navigate to the `processingData` folder:

    ```shell
    cd processingData
    ```

2. Use the following command to generate the image dataset:

    ```shell
    python visualize.py --blk_width <blk_width> --img_type <img_type>
    ```

    where:
    - `blk_width` defaults to 4, and can be set to 2, 4, 6, 8, or 10.
    - `img_type` defaults to `tsbt`, and can be set to `token`, `vsbt`, or `tvsbt`.

    For example, to generate an image dataset with block width 4 and type `tsbt`, use the following command:

    ```shell
    python visualize.py --blk_width 4 --img_type tsbt
    ```

## Train Model

1. Navigate to the `main` folder:

    ```shell
    cd main
    ```

2. Use the following command to train the model:

    ```shell
    python train_CPDP.py --dataset_path <dataset_path> --save_path <save_path>
    ```
    or
    ```shell
    python train_WPDP.py --dataset_path <dataset_path> --save_path <save_path>
    ```

    where:
    - `dataset_path` is the path to the generated image dataset.
    - `save_path` is the path where the training results will be saved.

    For example, to use the dataset at `../data/img_tsbt_4w` and save the results as `result_CPDP_tsbt_4w.xlsx`, use the following command:

    ```shell
    python train_CPDP.py --dataset_path ../data/img_tsbt_4w --save_path result_CPDP_tsbt_4w.xlsx
    ```
    or
    ```shell
    python train_WPDP.py --dataset_path ../data/img_tsbt_4w --save_path result_CPDP_tsbt_4w.xlsx
    ```

## Comprehensive Example

Suppose you want to generate an image dataset with block width 6 and type `vsbt`, and train using `train_WPDP.py`, the complete steps are as follows:

1. Navigate to the `processingData` folder and generate the dataset:

    ```shell
    cd processingData
    python visualize.py --blk_width 6 --img_type vsbt
    ```

2. Navigate to the `main` folder and train the model:

    ```shell
    cd main
    python train_WPDP.py --dataset_path ../data/img_vsbt_6w --save_path result_WPDP_vsbt_6w.xlsx
    ```

By following these steps, you can flexibly generate image datasets and train models according to your needs.
