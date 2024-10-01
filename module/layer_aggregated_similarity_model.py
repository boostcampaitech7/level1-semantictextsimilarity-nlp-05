import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from transformers import AutoConfig

from module.loss import CombinedL1PearsonLoss, CombinedMSEPearsonLoss

LOSS_FUNCTION = {
    "MSE": torch.nn.MSELoss(),
    "L1": torch.nn.L1Loss(),
    "Huber": torch.nn.HuberLoss(),
    "CMPLoss": CombinedMSEPearsonLoss(alpha=0.7),
    "CLPLoss": CombinedL1PearsonLoss(alpha=0.7),
}


class LayerAggregatedSimilarityModel(pl.LightningModule):
    def __init__(self, model_name, lr, loss_function):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.fine_tuning = False

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )

        # output_hidden_states 활성화 설정
        self.plm.config.output_hidden_states = True

        self.loss_func = LOSS_FUNCTION[loss_function]

        # klue/roberta-large의 레이어 수 및 hidden_size
        num_layers = 24  # klue/roberta-large의 레이어 수
        hidden_size = self.plm.config.hidden_size  # hidden state의 크기

        # CNN 레이어를 사용해 가중합을 할 수 있도록 설정 (초기화를 pooler에만 1, 나머지는 0으로 설정)
        self.cnn = torch.nn.Conv1d(in_channels=num_layers + 1, out_channels=1, kernel_size=1)

        # CNN 가중치 초기화 - pooler에 대응되는 값만 1로, 나머지는 0으로 설정
        # with torch.no_grad():
        # self.cnn.weight.zero_()  # 모든 가중치를 0으로 초기화
        # self.cnn.weight.normal_(mean=0.0, std=0.01)  # 모든 가중치를 작은 랜덤 값으로 초기화
        # self.cnn.weight[0, -1, 0] = 1  # pooler에 대응되는 마지막 채널의 가중치만 1로 설정
        # 가중치 합이 1이 되도록 조정
        # self.cnn.weight.data = F.normalize(self.cnn.weight.data, p=1, dim=1)  # dim=1을 기준으로 정규화하여 합이 1이 되도록 설정

        # bias는 0으로 초기화
        # torch.nn.init.zeros_(self.cnn.bias)

        # CNN 레이어를 Freeze 시키기
        for param in self.cnn.parameters():
            param.requires_grad = True

        # Encoder를 Freeze 시키고, CNN 및 기존 Linear Layer만 학습하도록 설정
        for param in self.plm.roberta.parameters():
            param.requires_grad = True

        # 기존 linear layer도 Freeze
        for param in self.plm.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.plm.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        pooler_output = hidden_states[-1][:, 0, :]  # 마지막 레이어의 [CLS] 토큰을 pooler output처럼 사용

        # Attention mask를 확장하여 padding 토큰 제외
        mask_expanded = attention_mask.unsqueeze(-1).expand(-1, -1, self.plm.config.hidden_size).float()

        # 모든 hidden states를 하나의 텐서로 병합하여 병렬 처리
        hidden_states_tensor = torch.stack(
            hidden_states[:-1], dim=1
        )  # shape: (batch_size, num_layers, seq_len, hidden_size)

        # mask 적용
        masked_hidden_states = hidden_states_tensor * mask_expanded.unsqueeze(
            1
        )  # (batch_size, num_layers, seq_len, hidden_size)

        # 각 레이어별 임베딩 합과 토큰 개수 계산
        sum_embeddings = masked_hidden_states.sum(dim=2)  # shape: (batch_size, num_layers, hidden_size)
        total_tokens = mask_expanded.sum(dim=1, keepdim=True)  # shape: (batch_size, 1, 1)

        # 0으로 나누는 것을 방지하기 위해 작은 값을 더함
        total_tokens = total_tokens + 1e-10

        # 패딩을 제외한 평균 임베딩 계산
        mean_layer_embeddings = sum_embeddings / total_tokens  # shape: (batch_size, num_layers, hidden_size)

        # pooler_output을 추가하여 총 25개의 벡터 생성
        combined_embeddings = torch.cat((mean_layer_embeddings, pooler_output.unsqueeze(1)), dim=1)

        # CNN을 통해 가중합을 수행
        combined_weighted_embedding = self.cnn(combined_embeddings)  # shape: (batch_size, 1, hidden_size)

        # 기존의 학습된 Linear Layer에 넣어 최종 유사도 계산
        similarity_score = self.plm.classifier(combined_weighted_embedding)

        return {"similarity_score": similarity_score, "combined_embedding": combined_weighted_embedding.squeeze(1)}

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        y = batch["labels"]
        similarity_score = self(input_ids=input_ids, attention_mask=attention_mask)["similarity_score"]
        loss = self.loss_func(similarity_score, y.float())
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        y = batch["labels"]
        similarity_score = self(input_ids=input_ids, attention_mask=attention_mask)["similarity_score"]
        loss = self.loss_func(similarity_score, y.float())
        self.log("val_loss", loss)
        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(similarity_score.squeeze(), y.squeeze()),
        )

        return loss

    def fine_tuning_for_pearson(self, lr=1e-5):
        # 마지막 레이어만 Pearson 상관계수에 맞게 미세조정하기 위한 함수
        for param in self.parameters():
            param.requires_grad = False  # 모든 파라미터 고정
        self.linear_layer.weight.requires_grad = True
        self.linear_layer.bias.requires_grad = True

        # 새로운 optimizer와 scheduler 설정
        optimizer = torch.optim.AdamW(self.linear_layer.parameters(), lr=lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        y = batch["labels"]
        similarity_score = self(input_ids=input_ids, attention_mask=attention_mask)["similarity_score"]

        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(similarity_score.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        similarity_score = self(input_ids=input_ids, attention_mask=attention_mask)["similarity_score"]

        return similarity_score.squeeze()

    def configure_optimizers(self):
        if self.fine_tuning:
            optimizer = self.fine_tuning_for_pearson(lr=1e-5)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=1, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
