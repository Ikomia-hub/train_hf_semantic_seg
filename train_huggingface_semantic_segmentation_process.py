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
from detectron2.layers import FrozenBatchNorm2d
import datasets
from datasets import Image, Dataset
import torch
from torch import nn
from transformers import Trainer,TrainerCallback, TrainingArguments,\
                         AutoFeatureExtractor, AutoModelForSemanticSegmentation
from transformers.integrations import MLflowCallback
import evaluate
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import yaml
import json
import copy
import mlflow


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainHuggingfaceSemanticSegmentationParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        self.cfg["model_name"] = "Segformer: nvidia/mit-b0"
        self.cfg["model_card"] = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.cfg["epochs"] = 50
        self.cfg["batch_size"] = 4
        self.cfg["imgsz"] = 224
        self.cfg["learning_rate"] = 0.00006
        self.cfg["test_percentage"] = 0.2
        self.cfg["expertModeCfg"] = ""
        self.cfg["output_folder"] = None
        self.cfg["expertModeCfg"] = None
        self.cfg["output_folder"] = None

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cfg["model_name"] = str(param_map["model_name"])
        self.cfg["model_card"] = str(param_map["model_card"])
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["imgsz"] = int(param_map["imgsz"])
        self.cfg["learning_rate"] = int(param_map["learning_rate"])
        self.cfg["test_percentage"] = float(param_map["test_percentage"])
        self.cfg["output_folder"] = str(param_map["output_folder"])
        self.cfg["expertModeCfg"] = param_map["expertModeCfg"]
        self.cfg["output_folder"] = param_map["output_folder"]

# --------------------
# - Class to handle stopping the train process on request
# --------------------
class StopTraining(TrainerCallback):

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        control.should_training_stop = True


class CustomMLflowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [MLflow](https://www.mlflow.org/). 
    Can be disabled by setting environment variable 
    `DISABLE_MLFLOW_INTEGRATION = TRUE`.
    """
    def __init__(self):
        self._initialized = False
        self._auto_end_run = False
        self._log_artifacts = False
        self._ml_flow = mlflow

    def setup(self, args, state, model):
        self._initialized = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            metrics = {}
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
            self._ml_flow.log_metrics(metrics=metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            if self._auto_end_run and self._ml_flow.active_run():
                self._ml_flow.end_run()

    def on_save(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero and self._log_artifacts:
            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            self._ml_flow.pyfunc.log_model(
                ckpt_dir,
                artifacts={"model_path": artifact_path},
                python_model=self._ml_flow.pyfunc.PythonModel(),
            )

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if (
            self._auto_end_run
            and callable(getattr(self._ml_flow, "active_run", None))
            and self._ml_flow.active_run() is not None
        ):
            self._ml_flow.end_run()

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
        self.num_labels = None
        self.id2label = None
        self.label2id = None
        self.trainer = None
        self.ignore_idx_eval = 255
        self.metric = None
        self.model_id = None
        self.training_args = None
        self.output_folder = None
        self.enableTensorboard(True)
        self.imgsz_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                "config", "model_img_size.json")
        self.config_to_remove = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                "config", "training_args_to_remove.yaml")

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.getParam()
        if param is not None:
            return param.cfg["epochs"]
        else:
            return 1

    def transforms(self, example_batch):
        images = [x.convert("RGB") for x in example_batch['pixel_values']]
        labels = [x for x in example_batch['label']]
        inputs = self.feature_extractor(images, labels)
        return inputs

    def compute_metrics(self, eval_pred):
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
            metrics = self.metric._compute(
                                        predictions=pred_labels,
                                        references=labels,
                                        num_labels=self.num_labels,
                                        ignore_index=self.ignore_idx_eval,
                                        reduce_labels=False
                                        )
            del metrics['per_category_iou']
            del metrics['per_category_accuracy']
        # add per category metrics as individual key-value pairs
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        return metrics

    def freeze_batchnorm2d(self, module: torch.nn.Module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                child: torch.nn.BatchNorm2d = child
                setattr(module, name, FrozenBatchNorm2d(child.num_features))
            else:
                self.freeze_batchnorm2d(child)
    
    def save_advanced_config(self, arg_dict):
        # list training arguments to dict
        arg_dict = arg_dict.to_sanitized_dict()

        # load unused training arguments
        with open(self.config_to_remove, 'r') as fp:
            unused_train_args = yaml.load(fp, Loader=yaml.FullLoader)
        unused_train_args_list = []
        for key, value in unused_train_args.items():
            unused_train_args_list.append(key)

        # remove unused args from training arguments
        arg_dict = dict([(key, val) for key, val in arg_dict.items() if key not in unused_train_args_list])

        # Add key to training arguments
        arg_dict = {"training_arg" : arg_dict}

        # Edit report to mlflow
        arg_dict["training_arg"]["report_to"] = "mlflow"

        # Add model id
        arg_dict["_name_or_path"] = self.model_id

        # Save training arguments
        output_file = os.path.join(self.output_folder, "advanced_config.yaml")
        with open(output_file, 'w') as outfile:
            yaml.dump(arg_dict, outfile)

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        # Mlflow setting
        os.environ["MLFLOW_FLATTEN_PARAMS"] = "TRUE"

        # Get input
        input = self.getInput(0)

        param = self.getParam()

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
        dataset = dataset.train_test_split(param.cfg["test_percentage"]) # Train/test split

        train_ds = dataset["train"]
        test_ds = dataset["test"]

        # Model name selection
        if param.cfg["expertModeCfg"] is None:
            if param.cfg["model_name"] == "From: Costum model name":
                self.model_id = param.cfg["model_card"]
            else:
                self.model_id = param.cfg["model_name"].split(": ",1)[1]
                param.cfg["model_card"] = None
        else:
            with open(param.cfg["expertModeCfg"]) as f:
                config = yaml.full_load(f)
                self.model_id = config["_name_or_path"]

        # Checking if the selected image size fits the model
        with open(self.imgsz_file, "r") as f:
            model_size_list = json.load(f)
        if self.model_id in model_size_list.keys():
            img_size = model_size_list[self.model_id]
            print(f'Image size parameter changed to ({img_size}x{img_size}) to match {self.model_id} model')
        else:
            img_size = param.cfg["imgsz"]

        # Image transformation (tensor, data augmentation) on-the-fly batches
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                                                                    self.model_id,
                                                                    size=img_size,
                                                                    return_tensors="pt",
                                                                    reduce_labels=False,
                                                                    resample=0,
                                                                    )

        train_ds.set_transform(self.transforms)
        test_ds.set_transform(self.transforms)

        # Labels preparation
        self.num_labels = len(input.data["metadata"]['category_names'])
        self.id2label = input.data["metadata"]['category_names']
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Loading Model
        model = AutoModelForSemanticSegmentation.from_pretrained(
                                                            self.model_id,
                                                            num_labels = self.num_labels,
                                                            id2label = self.id2label,
                                                            label2id = self.label2id,
                                                            ignore_mismatched_sizes=True
                                                            )
        model.train()

        # Tensorboard directory
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        tb_dir = str((Path(core.config.main_cfg["tensorboard"]["log_uri"]) / str_datetime))

        # Setting up output directory
        if param.cfg["output_folder"] is None:
            param.cfg["output_folder"] = os.path.join(os.path.dirname(os.path.realpath(__file__)),\
                                         "outputs", self.model_id, str_datetime)
        os.makedirs(param.cfg["output_folder"], exist_ok=True)
        self.output_folder = param.cfg["output_folder"]

        # Checking batch size
        if "nvidia" not in self.model_id:
            if param.cfg["expertModeCfg"] is None: 
                if param.cfg["batch_size"] == 1:
                    self.freeze_batchnorm2d(model)
            else:
                if config["training_arg"]["per_device_train_batch_size"] == 1 :
                    self.freeze_batchnorm2d(model)

        # Hyperparameters and costumization settings during training
        if param.cfg["expertModeCfg"] is None:
            self.training_args = TrainingArguments(
                param.cfg["output_folder"],
                learning_rate=param.cfg["learning_rate"],
                num_train_epochs=param.cfg["epochs"],
                per_device_train_batch_size=param.cfg["batch_size"],
                per_device_eval_batch_size=param.cfg["batch_size"],
                evaluation_strategy="steps",
                save_strategy="steps",
                save_steps=500,
                save_total_limit=1,
                eval_steps=500,
                logging_steps=1,
                eval_accumulation_steps=5,
                load_best_model_at_end=False,
                logging_dir=tb_dir,
                remove_unused_columns=False,
                report_to=None,
                dataloader_drop_last=True,
                prediction_loss_only =False,
            )
        else:
            print("Loading training arguments from yaml file")
            args = config["training_arg"]
            self.training_args = TrainingArguments(
                    param.cfg["output_folder"],
                    **args)

        self.metric = evaluate.load("mean_iou")

        # Instantiation of the Trainer API for training with Pytorch
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=self.compute_metrics,
            callbacks = [CustomMLflowCallback]
        )
        # Remove default mlflow callback
        self.trainer.remove_callback(MLflowCallback)

        self.beginTaskRun()

        # Start training loop
        self.trainer.train()

        # Save advanced config
        self.save_advanced_config(self.training_args)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun
        self.endTaskRun()

    def stop(self):
        super().stop()
        print("Stopping requested...")
        self.trainer.add_callback(StopTraining)
        print("Saving model...")
        self.trainer.save_model()
        self.trainer.save_state()
        self.save_advanced_config(self.training_args)
        print("advanced_config.yaml saved")
        print("Model saved.")


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
