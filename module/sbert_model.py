import pytorch_lightning as pl
import torch
import torchmetrics
import transformers
from tqdm.auto import tqdm


class SBERTModel(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name,
            output_hidden_states=True,  # 모든 레이어의 hidden states를 반환하도록 설정
        )

        # 모든 레이어를 우선 freeze
        for param in self.plm.parameters():
            param.requires_grad = False

        # 특정 레이어만 fine-tuning을 위해 unfreeze (예: 마지막 3개 레이어)
        for param in self.plm.encoder.layer[-3:].parameters():
            param.requires_grad = True

        # Loss 계산을 위해 사용될 MSELoss를 호출합니다.
        self.loss_func = torch.nn.MSELoss()

    def forward(self, batch):
        input_ids_1 = batch["input_ids_1"]
        attention_mask_1 = batch["attention_mask_1"]
        input_ids_2 = batch["input_ids_2"]
        attention_mask_2 = batch["attention_mask_2"]

        # # 모든 레이어의 hidden states 가져오기
        # outputs_1 = self.plm(input_ids=input_ids_1, attention_mask=attention_mask_1, output_hidden_states=True)
        # outputs_2 = self.plm(input_ids=input_ids_2, attention_mask=attention_mask_2, output_hidden_states=True)

        # # 여러 레이어의 [CLS] 토큰 벡터를 결합 (예: 마지막 4개 레이어)
        # cls_embeddings_1 = torch.cat([outputs_1.hidden_states[-i][:, 0, :] for i in range(1, 5)], dim=-1)
        # cls_embeddings_2 = torch.cat([outputs_2.hidden_states[-i][:, 0, :] for i in range(1, 5)], dim=-1)

        embedding_1 = self.plm(input_ids=input_ids_1, attention_mask=attention_mask_1)["pooler_output"]
        embedding_2 = self.plm(input_ids=input_ids_2, attention_mask=attention_mask_2)["pooler_output"]

        # 코사인 유사도 계산
        similarity = torch.nn.functional.cosine_similarity(embedding_1, embedding_2)
        return similarity

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        y = batch["labels"]  # 정답 레이블 추출

        # Scale을 맞추기 위함인데, 이론적으로 logits(코사인 유사도)는 범위가 -1~1이지만
        # 학습을 거치면 두 문장이 비슷할수록 1에 가까워지고 상관없으면 0에 가까워져
        # 음수가 나오는 일이 거의 없으므로 코사인 유사도를 0~1의 범위로 보고 5배 Scaling한다.
        loss = self.loss_func(5 * logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        y = batch["labels"]
        loss = self.loss_func(5 * logits, y.float())
        self.log("val_loss", loss)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(5 * logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        y = batch["labels"]

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(5 * logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        logits = self(batch)

        return 5 * logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
