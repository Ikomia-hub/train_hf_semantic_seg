<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/train_hf_semantic_seg/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">train_hf_semantic_seg</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_hf_semantic_seg">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_hf_semantic_seg">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_hf_semantic_seg/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_hf_semantic_seg.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Train on semantic segmentation models available on Hugging Face (Segformer, BeiT, Data2Vec-vision).

![Segformer output](https://raw.githubusercontent.com/Ikomia-hub/train_hf_semantic_seg/feat/new_readme/icons/output.jpg)


## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add data loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "semantic_segmentation",
}) 

# Add train algorithm 
train = wf.add_task(name="train_hf_semantic_seg", auto_connect=True)

# Launch your training on your data
wf.run()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_card** (int) - default 'nvidia/segformer-b0-finetuned-ade-512-512': Name of the model.
- **batch_size** (int) - default '4': Number of samples processed before the model is updated.
- **epochs** (int) - default '50': Number of complete passes through the training dataset.
- **input_size** (int) - default '224': Size of the input image.
- **learning_rate** (float) - default '0.00006': Step size at which the model's parameters are updated during training.
- **dataset_split_ratio** (float) – default '0.9': Divide the dataset into train and evaluation sets ]0, 1[.
- **output_folder** (str, *optional*): path to where the model will be saved. 
- **config_file** (str, *optional*): path to the training config file .yaml. 


**Parameters** should be in **strings format**  when added to the dictionary.


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add data loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "semantic_segmentation",
}) 

# Add train algorithm 
train = wf.add_task(name="train_hf_semantic_seg", auto_connect=True)
train.set_parameters({
    "model_card": "nvidia/mit-b2",
    "batch_size": "4",
    "epochs": "50",
    "learning_rate": "0.00006",
    "dataset_split_ratio": "0.8",
}) 

# Launch your training on your data
wf.run()
```


## :fast_forward: Advanced usage 

This algorithm proposes to fine-tune semantic segmentation models available on Hugging Face:

1. **[BEiT](https://huggingface.co/docs/transformers/model_doc/beit)** (from Microsoft) released with the paper [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254) by Hangbo Bao, Li Dong, Furu Wei.
    - [microsoft/beit-base-patch16-224-pt22k](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k)
    - microsoft/beit-base-patch16-224
    - microsoft/beit-base-patch16-384
    - microsoft/beit-large-patch16-224-pt22k
    - microsoft/beit-large-patch16-224
    - microsoft/beit-large-patch16-384
    - microsoft/beit-large-patch16-512


1. **[Data2Vec](https://huggingface.co/docs/transformers/model_doc/data2vec)** (from Facebook) released with the paper [Data2Vec:  A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/abs/2202.03555) by Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli.
    - [facebook/data2vec-vision-base](https://huggingface.co/facebook/data2vec-vision-base)



1. **[SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer)** (from NVIDIA) released with the paper [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) by Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo.
    - [nvidia/mit-b0](https://huggingface.co/nvidia/mit-b0) 
    - nvidia/mit-b1
    - nvidia/mit-b2
    - nvidia/mit-b3
    - nvidia/mit-b4
    - nvidia/mit-b5
