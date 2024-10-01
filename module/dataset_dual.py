import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs_1, inputs_2, targets=[]):
        self.inputs_1 = inputs_1
        self.inputs_2 = inputs_2
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        # 딕셔너리로 반환해야 DataCollatorWithPadding의 입력으로 사용 가능
        if len(self.targets) == 0:
            return {
                "input_ids_1": torch.tensor(self.inputs_1[idx]),  # input_ids_1 반환
                "input_ids_2": torch.tensor(self.inputs_2[idx]),  # input_ids_2 반환
            }
        else:
            return {
                "input_ids_1": torch.tensor(self.inputs_1[idx]),  # input_ids_1 반환
                "input_ids_2": torch.tensor(self.inputs_2[idx]),  # input_ids_2 반환
                "labels": torch.tensor(self.targets[idx], dtype=torch.float),  # 정답 레이블
            }

    def __len__(self):
        return len(self.inputs_1)


class DualSentenceDataCollatorWithPadding:
    # 원래 DataCollatorWithPadding은 한 문장에 대해서만 attention_mask를 반환하기 때문에 두 문장에 대해 각각 attention_mask를 반환하도록 변경
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, batch):
        # input_ids_1과 input_ids_2를 분리
        input_ids_1 = [item["input_ids_1"] for item in batch]
        input_ids_2 = [item["input_ids_2"] for item in batch]

        # Collator를 통해 input_ids_1과 input_ids_2 각각에 대해 attention_mask를 생성
        batch_1 = self.data_collator([{"input_ids": ids} for ids in input_ids_1])
        batch_2 = self.data_collator([{"input_ids": ids} for ids in input_ids_2])

        # 결과를 합쳐서 최종 batch 생성
        combined_batch = {
            "input_ids_1": batch_1["input_ids"],
            "attention_mask_1": batch_1["attention_mask"],
            "input_ids_2": batch_2["input_ids"],
            "attention_mask_2": batch_2["attention_mask"],
        }

        # 만약 레이블(label)이 존재하는 경우 추가
        if "labels" in batch[0]:
            labels = torch.tensor([item["labels"] for item in batch], dtype=torch.float)
            combined_batch["labels"] = labels

        return combined_batch


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.data_collator = DualSentenceDataCollatorWithPadding(tokenizer=self.tokenizer)

        self.target_columns = ["label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence_1", "sentence_2"]

    def tokenizing(self, dataframe):
        data_1 = []
        data_2 = []
        for _, item in tqdm(dataframe.iterrows(), desc="tokenizing", total=len(dataframe)):
            # 두 문장을 각각 토큰화합니다.
            text_1 = item["sentence_1"]
            text_2 = item["sentence_2"]
            outputs_1 = self.tokenizer(text_1, add_special_tokens=True, padding=False, truncation=True)
            outputs_2 = self.tokenizer(text_2, add_special_tokens=True, padding=False, truncation=True)
            data_1.append(outputs_1["input_ids"])
            data_2.append(outputs_2["input_ids"])
        return data_1, data_2

    def token_add(self, tokens: list) -> None:
        current_vocab = set(self.tokenizer.vocab.keys())

        new_tokens = [token for token in tokens if token not in current_vocab]

        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs_1, inputs_2 = self.tokenizing(data)

        return inputs_1, inputs_2, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습 데이터 준비
            train_inputs_1, train_inputs_2, train_targets = self.preprocessing(train_data)

            # 검증 데이터 준비
            val_inputs_1, val_inputs_2, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs_1, train_inputs_2, train_targets)
            self.val_dataset = Dataset(val_inputs_1, val_inputs_2, val_targets)
        else:
            # 평가 데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs_1, test_inputs_2, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs_1, test_inputs_2, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs_1, predict_inputs_2, _ = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs_1, predict_inputs_2)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.data_collator
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.batch_size, collate_fn=self.data_collator
        )
