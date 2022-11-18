# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.core.task import TaskParam
from ikomia.dnn import datasetio, dnntrain
import copy
# Your imports below
import datasets
from datasets import Image, Dataset
from torchvision.transforms import ColorJitter
import torch
from torch import nn
from transformers import Trainer,TrainerCallback, TrainingArguments,\
                         AutoFeatureExtractor, AutoModelForSemanticSegmentation
import evaluate
import numpy as np
import os
from datetime import datetime
from pathlib import Path


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainHuggingfaceSemanticSegmentationParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        self.cfg["model_name"] = "nvidia/mit-b0"
        self.cfg["epochs"] = 50
        self.cfg["batch_size"] = 4
        self.cfg["imgsz"] = 512
        self.cfg["learning_rate"] = 0.00006
        self.cfg["save_steps"] = 20
        self.cfg["eval_steps"] = 20
        self.cfg["eval_accumulation_steps"] = 5
        self.cfg["test_percent"] = 0.2
        self.cfg["output_folder"] = None
        self.cfg["ignore_idx_eval"] = 0
        self.cfg["output_folder"] = None

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cfg["model_name"] = str(param_map["model_name"])
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["imgsz"] = int(param_map["imgsz"])
        self.cfg["learning_rate"] = int(param_map["learning_rate"])
        self.cfg["save_steps"] = int(param_map["save_steps"])
        self.cfg["eval_steps"] = int(param_map["eval_steps"])
        self.cfg["eval_accumulation_steps"] = int(param_map["eval_accumulation_steps"])
        self.cfg["test_percent"] = float(param_map["test_percent"])
        self.cfg["output_folder"] = str(param_map["output_folder"])
        self.cfg["ignore_idx_eval"] = int(param_map["ignore_idx_eval"])
        self.cfg["output_folder"] = param_map["output_folder"]

# --------------------
# - Class to handle stopping the train process on request
# --------------------
class StopTraining(TrainerCallback):

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        control.should_training_stop = True


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainHuggingfaceSemanticSegmentation(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # Add input/output of the process here

        # Create parameters class
        if param is None:
            self.setParam(TrainHuggingfaceSemanticSegmentationParam())
        else:
            self.setParam(copy.deepcopy(param))

        self.feature_extractor = None
        self.jitter = None
        self.num_labels = None
        self.trainer = None
        self.ignore_idx_eval = None
        self.metric = None
        self.enableTensorboard(True)

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.getParam()
        if param is not None:
            return param.cfg["epochs"]
        else:
            return 1

    def train_transforms(self, example_batch):
        #param = self.getParam()
        images = [self.jitter(x) for x in example_batch['pixel_values']] # Data augmentation
        labels = [x for x in example_batch['label']]
        inputs = self.feature_extractor(images, labels)
        return inputs

    def val_transforms(self, example_batch):
        #param = self.getParam()
        images = [x for x in example_batch['pixel_values']]
        labels = [x for x in example_batch['label']]
        inputs = self.feature_extractor(images, labels)
        return inputs

    def compute_metrics(self, eval_pred):
        #metric = evaluate.load("mean_iou")
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            # scale the logits to the size of the label
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size = labels.shape[-2:],
                mode = "bilinear",
                align_corners = False,
            ).argmax(dim = 1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = self.metric.compute(
                                        predictions=pred_labels,
                                        references=labels,
                                        num_labels=self.num_labels,
                                        ignore_index=self.ignore_idx_eval,
                                        reduce_labels=self.feature_extractor.reduce_labels
                                        )

            for key, value in metrics.items():
                if type(value) is np.ndarray:
                    metrics[key] = value.tolist()
            return metrics

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        input = self.getInput(0)

        param = self.getParam()

        self.beginTaskRun()

        # Dataset preparation and train/test split
        filename_list = []
        for image in input.data["images"]:
            filename_list.append(image["filename"])
        dict_img = {'pixel_values': filename_list}
        dataset_img = Dataset.from_dict(dict_img).cast_column("pixel_values", Image()) # Images dataset

        semantic_seg_masks_file = []
        for image in input.data["images"]:
            semantic_seg_masks_file.append(image["semantic_seg_masks_file"])
        dict_mask = {'label': semantic_seg_masks_file}
        dataset_mask = Dataset.from_dict(dict_mask).cast_column("label", Image()) # Mask dataset

        # Merging images and masks
        dataset = datasets.concatenate_datasets([dataset_img, dataset_mask], axis=1)
        dataset = dataset.shuffle(seed=1)
        dataset = dataset.train_test_split(param.cfg["test_percent"]) # Train/test split

        train_ds = dataset["train"]
        test_ds = dataset["test"]

        # Selection of the feature extractor
        model_feature_extractor = param.cfg["model_name"]

        # Selection of the pre-trained model
        pretrained_model_name  = param.cfg["model_name"]

        # Image transformation (tensor, data augmentation) on-the-fly batches
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                                                                    model_feature_extractor,
                                                                    size = param.cfg["imgsz"]
                                                                    )
        self.jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

        train_ds.set_transform(self.train_transforms)
        test_ds.set_transform(self.val_transforms)

        # Labels preparation
        self.num_labels = len(input.data["metadata"]['category_names'])
        id2label = input.data["metadata"]['category_names']
        label2id = {v: k for k, v in id2label.items()}

        # Index to be ignored during evaluation
        self.ignore_idx_eval = param.cfg["ignore_idx_eval"]

        # Loading Model
        model = AutoModelForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels = self.num_labels,
            id2label = id2label,
            label2id = label2id,
            ignore_mismatched_sizes=True)

        # Setting up output directory
        if param.cfg["output_folder"] is None:
            param.cfg["output_folder"] = os.path.join(os.path.dirname(os.path.realpath(__file__)),\
                                         "outputs", param.cfg["model_name"])
        os.makedirs(param.cfg["output_folder"], exist_ok=True)

        # Tensorboard directory
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        tb_dir = str((Path(core.config.main_cfg["tensorboard"]["log_uri"]) / str_datetime))

        # Hyperparameters and costumization settings during training
        training_args = TrainingArguments(
            param.cfg["output_folder"],
            learning_rate=param.cfg["learning_rate"],
            num_train_epochs=param.cfg["epochs"],
            per_device_train_batch_size=param.cfg["batch_size"],
            per_device_eval_batch_size=param.cfg["batch_size"],
            save_total_limit=3,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=param.cfg["save_steps"],
            eval_steps=param.cfg["eval_steps"],
            logging_steps=1,
            eval_accumulation_steps=param.cfg["eval_accumulation_steps"],
            load_best_model_at_end=True,
            remove_unused_columns=False,
            logging_dir=tb_dir,
            report_to="mlflow"
        )

        self.metric = evaluate.load("mean_iou")

        # Instantiation of the Trainer API for training with Pytorch
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

        self.trainer.save_model(output_dir=param.cfg["output_folder"])

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun
        self.endTaskRun()

    def stop(self):
        super().stop()
        print("Stopping requested...")
        self.trainer.add_callback(StopTraining)


# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainHuggingfaceSemanticSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_huggingface_semantic_segmentation"
        self.info.shortDescription = "Train models for semantic segmentation"\
                                     "with transformers from HuggingFace."
        self.info.description = "This model proposes train on semantic segmentation"\
                                "using pre-trained models available on Hugging Face."

        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/icon.png"
        self.info.authors = "Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond,"\
                            "Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault,"\
                            "Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer,"\
                            "Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu,"\
                            "Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame,"\
                            "Quentin Lhoest, Alexander M. Rush"
        self.info.article = "Huggingface's Transformers: State-of-the-art Natural Language Processing"
        self.info.journal = "EMNLP"
        self.info.license = "Apache License Version 2.0"
        # URL of documentation
        self.info.documentationLink = "https://www.aclweb.org/anthology/2020.emnlp-demos.6"
        # Code source repository
        self.info.repository = "https://github.com/huggingface/transformers"
        # Keywords used for search
        self.info.keywords = "semantic, segmentation, transformer, encoder MLP, decoder,"\
                            "Hugging Face, Pytorch, Segformer, DPT, Beit, data2vec"

    def create(self, param=None):
        # Create process object
        return TrainHuggingfaceSemanticSegmentation(self.info.name, param)
