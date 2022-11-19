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
from ikomia.utils import pyqtutils, qtconversion
from train_huggingface_semantic_segmentation.train_huggingface_semantic_segmentation_process import TrainHuggingfaceSemanticSegmentationParam
import json
import os 
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class TrainHuggingfaceSemanticSegmentationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = TrainHuggingfaceSemanticSegmentationParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_Layout = QGridLayout()

        # Model name
        self.combo_model = pyqtutils.append_combo(self.grid_Layout, "Model name")
        model_list_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "config", "model_list.json")
        with open(model_list_path, "r") as f:
            model_name_list = json.load(f)

        for model_name in model_name_list["models"]:
            self.combo_model.addItem(model_name)

        self.combo_model.setCurrentText(self.parameters.cfg["model_name"])

        # Epochs
        self.spin_epochs = pyqtutils.append_spin(self.grid_Layout, "Epochs",
                                                self.parameters.cfg["epochs"])

        # Batch size
        self.spin_batch = pyqtutils.append_spin(self.grid_Layout, "Batch size",
                                                self.parameters.cfg["batch_size"])

        # Input size
        self.spin_train_imgsz = pyqtutils.append_spin(self.grid_Layout, "Image size",
                                                    self.parameters.cfg["imgsz"])
        # Learning rate
        self.spin_lr = pyqtutils.append_double_spin(self.grid_Layout, "Learning rate",
                                                    self.parameters.cfg["learning_rate"],
                                                    min = 0.00001, max = 0.10000,
                                                    step = 0.00001, decimals = 5)

        # Train test split
        self.spin_train_test_split = pyqtutils.append_double_spin(self.grid_Layout,
                                                                "Test image percentage",
                                                                self.parameters.cfg["test_percentage"],
                                                                min = 0.1, max = 1.0,
                                                                step = 0.1, decimals = 1)

        # Ingore index
        self.ignore_idx_eval = pyqtutils.append_double_spin(self.grid_Layout, 
                                                            "Index number ignored during eval",
                                                            self.parameters.cfg["ignore_idx_eval"],
                                                            min = 0, max = 250, step = 1, decimals = 0)

        # Output folder
        self.browse_folder = pyqtutils.append_browse_file(self.grid_Layout, label="Output folder",
                                                        path=self.parameters.cfg["output_folder"],
                                                        tooltip="Select output folder",
                                                        mode=QFileDialog.Directory)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_Layout)

        # Set widget layout
        self.setLayout(layout_ptr)

    def onApply(self):
        # Apply button clicked slot
        # Get parameters from widget
        # Send signal to launch the process
        self.parameters.cfg["model_name"] = self.combo_model.currentText()
        self.parameters.cfg["epochs"] = self.spin_epochs.value()
        self.parameters.cfg["batch_size"] = self.spin_batch.value()
        self.parameters.cfg["learning_rate"] = self.spin_lr.value()
        self.parameters.cfg["test_percentage"] = self.spin_train_test_split.value()
        self.parameters.cfg["imgsz"] = self.spin_train_imgsz.value()
        self.parameters.cfg["ignore_idx_eval"] = self.ignore_idx_eval.value()
        self.parameters.cfg["output_folder"] = self.browse_folder.path
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class TrainHuggingfaceSemanticSegmentationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "train_huggingface_semantic_segmentation"

    def create(self, param):
        # Create widget object
        return TrainHuggingfaceSemanticSegmentationWidget(param, None)
