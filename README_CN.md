# 生成图片数据集和训练模型

[English](README.md)

## 设置 Conda 环境

### 修改 `TreeCV.yml` 文件

要设置 Conda 环境，请对 `TreeCV.yml` 文件进行以下修改：

通过更新 `prefix` 字段来修改你要安装的路径。例如：

    ```yaml
    prefix: 你要安装的路径
    ```

### 创建 Conda 环境

修改 `TreeCV.yml` 文件后，使用以下命令创建新环境：

```shell
conda env create -f TreeCV.yml
```

## 生成图片数据集

1. 首先进入 `processingData` 文件夹：

    ```shell
    cd processingData
    ```

2. 使用以下命令生成图片数据集：

    ```shell
    python visualize.py --blk_width <blk_width> --img_type <img_type>
    ```

    其中：
    - `blk_width` 默认是 4，可以选择 2，4，6，8，10。
    - `img_type` 默认是 `tsbt`，可以选择 `token`, `vsbt`, `tvsbt`。

    例如，要生成块宽度为 4 类型为 `tsbt` 的图片数据集，使用以下命令：

    ```shell
    python visualize.py --blk_width 4 --img_type tsbt
    ```

## 训练模型

1. 进入 `main` 文件夹：

    ```shell
    cd main
    ```

2. 使用以下命令训练模型：

    ```shell
    python train_CPDP.py --dataset_path <dataset_path> --save_path <save_path>
    ```
    或
    ```shell
    python train_WPDP.py --dataset_path <dataset_path> --save_path <save_path>
    ```

    其中：
    - `dataset_path` 是生成的图片数据集的路径。
    - `save_path` 是模型训练结果保存的路径。

    例如，要使用路径为 `../data/img_tsbt_4w` 的数据集，并将结果保存为 `result_CPDP_tsbt_4w.xlsx`，使用以下命令：

    ```shell
    python train_CPDP.py --dataset_path ../data/img_tsbt_4w --save_path result_CPDP_tsbt_4w.xlsx
    ```
    或
    ```shell
    python train_WPDP.py --dataset_path ../data/img_tsbt_4w --save_path result_CPDP_tsbt_4w.xlsx
    ```

## 综合示例

假设要生成块宽度为 6 类型为 `vsbt` 的图片数据集，并使用 `train_WPDP.py` 进行训练，完整步骤如下：

1. 进入 `processingData` 文件夹并生成数据集：

    ```shell
    cd processingData
    python visualize.py --blk_width 6 --img_type vsbt
    ```

2. 进入 `main` 文件夹并训练模型：

    ```shell
    cd main
    python train_WPDP.py --dataset_path ../data/img_vsbt_6w --save_path result_WPDP_vsbt_6w.xlsx
    ```

通过这些步骤，可以根据需求灵活生成图片数据集并训练模型。
