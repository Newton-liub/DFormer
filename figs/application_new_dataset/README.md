# <p align=center>`How to apply DFormer to new dataset`</p>


> If there are some questions or suggestions, please raise an issue or contact us at bowenyin@mail.nankai.edu.cn.

Acknowledgment to [wuYwen1](https://github.com/wuYwen1), he apply the DFormer to his own dataset and discuss with us about how to achieve the application to new datasets. 
The segmentation results are shown in the below:

<p align="center">
    <img src="application.jpg" width="600"  width="1200"/> <br />
    <em> 
    </em>
</p>

## 1. 🌟  Prepare your own data

If your dataset not contain the label, it is suggested to use [labelme](https://github.com/wkentaro/labelme) to label the data.

We recommond to orginize the dataset as follows:


```shell
<../dataset>
|-- <DatasetName1>
    |-- <RGB>
        |-- <name1>.<ImageFormat>
        |-- <name2>.<ImageFormat>
        ...
    |-- <Depth>
        |-- <name1>.<DepthFormat>
    |-- <Label>
        |-- <name1>.<LabelFormat>
    |-- train.txt
    |-- test.txt
```

where the \<Format\> tends to be jpg or png, the two `.txt` files are used to index the training and evaluating sets. The maintained template defaults to the repository sibling directory `../dataset`; if you choose another root, change `C.root_dir` explicitly and keep the documentation and launch command consistent.
<!-- We provide a [template]() to generate the indexing files. -->
The format of the indexing files (test.txt, train.txt) are as follows:

```shell
RGB/file1_name.<format>
```

Note that we assume that the depth, RGB, and label for the same sample share the name and are in different folders.


> Requirement: Depth maps and labels are grayscale images. The color image channel order must be declared for the selected loader instead of inferred from the word “RGB”.
In labels, if 0 is background and should be ignored while 1–N are foreground classes, use `C.gt_transform = True`, `C.background = 255`, and `C.num_classes = N`. If background participates in loss and mIoU and labels are already `0..N`, use `C.gt_transform = False` and `C.num_classes = N + 1`. Do not delete or edit the shared transform in `utils/dataloader/RGBXDataset.py`, because that would silently change every dataset.


## 2. 🌟 Make your own config file

We provide a config template file at `local_configs\template\DFormer_Large.py'.

(1) replace the dataset_name with the name and folder name of your own data.

>C.dataset_name = 'dataset_name'  <br />
C.dataset_path = osp.join(C.root_dir, 'dataset_name')

(2) Change the format for rgb, depth and label.

>C.rgb_format = '.jpg' <br />
C.x_format = '.png' <br />
C.gt_format = '.png'

(3) Replace the example counts with positive integers from the final `train.txt` and `test.txt`. The template values `1` are import-safe examples, not real dataset statistics.

>C.num_train_imgs = 1595  # example only <br />
C.num_eval_imgs = 1576   # example only

(4) Replace the example class metadata with the real integer count and matching names.

>C.num_classes = 15  # example only <br />
C.class_names = ['class_1', '...']

(5) Change the training size.

>C.image_height = 480 <br />
C.image_width = 640

You can also tune some other parameters for better performance.

After replacing every example value, first verify that the config imports from the repository root:

```bash
python -c "import local_configs.template.DFormer_Large as c; print(c.config.dataset_name, c.config.num_train_imgs, c.config.num_classes)"
```

Then change the config path in a reviewed launch command. The root `.sh` files are upstream multi-GPU examples; inspect their fixed GPU count, dataset, checkpoint and `PYTHONPATH` before reuse.
