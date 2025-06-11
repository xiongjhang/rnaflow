import torch
from pytorch_lightning import LightningModule

from models.modules.celltrack_mpnn import Net_new_new


class CellTrackLitModel(LightningModule):

    def __init__(
        self,
        sample,
        weight_loss,
        directed,
        model_params,
        separate_models,
        # seg_model_params,
        loss_weights,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.separate_models = separate_models
        self.model = Net_new_new(**model_params)
        # loss function
        if self.hparams.one_hot_label:
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_weights))
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, edge_feat):
        return self.model(x, edge_index, edge_feat)







