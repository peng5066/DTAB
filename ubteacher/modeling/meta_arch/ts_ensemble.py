# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn as nn


"""
This class serves as an ensemble model, encapsulating both a teacher model and a student model.

Within the constructor, we begin by invoking the `nn.Module` constructor for initialization. Subsequently, we address potential parallel wrappers by examining if `modelTeacher` and `modelStudent` are instances of `DistributedDataParallel` or `DataParallel`.

If a parallel wrapper is detected, we retrieve the underlying, original model object via the `.module` attribute. Following this, we store the teacher model in `self.modelTeacher` and the student model in `self.modelStudent`.

Ultimately, the purpose of this class is to combine the teacher and student models into a unified ensemble, facilitating their joint management and utilization during model saving and loading procedures.
"""
class EnsembleTSModel(nn.Module):
    def __init__(self, modelTeacher, modelStudent):
        super(EnsembleTSModel, self).__init__()

        if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
            modelTeacher = modelTeacher.module
        # if isinstance(modelTeacher2, (DistributedDataParallel, DataParallel)):
        #     modelTeacher2 = modelTeacher2.module
        if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
            modelStudent = modelStudent.module

        self.modelTeacher  = modelTeacher
        # self.modelTeacher2 = modelTeacher2
        self.modelStudent  = modelStudent