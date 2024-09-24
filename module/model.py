from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from module.loss import CombinedMSEPearsonLoss, CombinedL1PearsonLoss

LOSS_FUNCTION = {
            'MSE': torch.nn.MSELoss(),
            'L1': torch.nn.L1Loss(),
            'Huber': torch.nn.HuberLoss(),
            'CMPLoss': CombinedMSEPearsonLoss(alpha=0.7),
            'CLPLoss': CombinedL1PearsonLoss(alpha=0.7),
        }

class Model(pl.LightningModule):
    def __init__(self, model_name, lr,loss_function):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        
        self.loss_func = LOSS_FUNCTION[loss_function]

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=1, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
    


class MultitaksModel(pl.LightningModule):
    def __init__(self,model_name:str, lr:float, loss_function:str , weight:float):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.encoder = transformers.AutoModel.from_pretrained(model_name)

        self.similarity_head = torch.nn.Linear(self.encoder.config.hidden_size,1)
        self.classification_head = torch.nn.Linear(self.encoder.config.hidden_size,2)

        self.similarity_loss = LOSS_FUNCTION[loss_function]
        self.classification_loss = torch.nn.CrossEntropyLoss()

        self.similarity_metric = torchmetrics.functional.pearson_corrcoef
        self.classification_metric = torchmetrics.Accuracy(task="binary")

        self.similarity_weight = weight
        self.classification_weight = 1 - weight

    def forward(self, x):
        x = self.encoder(x)['last_hidden_state']
        pooler_x = x[:,0 ,:]
        similarity = self.similarity_head(pooler_x)
        classification = self.classification_head(pooler_x)

        return {
            'similarity': similarity.squeeze(),
            'classification': classification
        }
    
    def training_step(self, batch, batch_idx):
        x, similarity_y,classification_y = batch
        #similarity_y = y[0]
        #classification_y = y[1]
        logits = self(x)
        similarity_loss = self.similarity_loss(logits['similarity'], similarity_y.float())
        classification_loss = self.classification_loss(logits['classification'], classification_y.long())
        loss = similarity_loss * self.similarity_weight + classification_loss * self.classification_weight
        
        self.log("similarity_train_loss", similarity_loss)
        self.log("classification_train_loss", classification_loss)
        self.log("total_train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, similarity_y,classification_y = batch
        #similarity_y = y[0]
        #classification_y = y[1]
        logits = self(x)
        classification_pred = torch.argmax(logits['classification'], dim=1)
        similarity_loss = self.similarity_loss(logits['similarity'], similarity_y.float())
        classification_loss = self.classification_loss(logits['classification'], classification_y.long())
        loss = similarity_loss * self.similarity_weight + classification_loss * self.classification_weight
        
        self.log("similarity_val_loss", similarity_loss)
        self.log("classification_val_loss", classification_loss)
        self.log("total_val_loss", loss)
        self.log("val_pearson", self.similarity_metric(logits['similarity'], similarity_y.squeeze()))
        self.log("classification_val_acc", self.classification_metric( classification_pred, classification_y))

        return loss
    
    def test_step(self, batch, batch_idx):
        x, similarity_y,classification_y = batch
        #similarit_y = y[0]
        #classification_y = y[1]

        logits = self(x)
        classification_pred = torch.argmax(logits['classification'], dim=1)
        similarity_loss = self.similarity_loss(logits['similarity'], similarity_y.float())
        classification_loss = self.classification_loss(logits['classification'], classification_y)
        loss = similarity_loss * self.similarity_weight + classification_loss * self.classification_weight

        self.log("similarity_test_pearson", self.similarity_metric(logits['similarity'], similarity_y.squeeze()))
        self.log("classification_test_acc", self.classification_metric(classification_pred, classification_y))

            
    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits['similarity']
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=1, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    
    
