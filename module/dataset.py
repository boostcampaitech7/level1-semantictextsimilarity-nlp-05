import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from tqdm.auto import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, inputs, segment_ids, attention_masks, targets, multi_task=False
    ):  #  parameter(segment_ids, attention_masks) 추가
        self.inputs = inputs
        self.segment_ids = segment_ids  # segment_ids를 __init__에서 정의
        self.attention_masks = attention_masks  # attention_masks를 __init__에서 정의
        self.targets = targets
        self.multi_task = multi_task

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            # attention_mask 추가
            return (
                torch.tensor(self.inputs[idx]),
                torch.tensor(self.segment_ids[idx]),
                torch.tensor(self.attention_masks[idx]),
            )
        else:
            if self.multi_task:
                return (
                    torch.tensor(self.inputs[idx]),
                    torch.tensor(self.segment_ids[idx]),  # segment_ids 추가
                    torch.tensor(self.attention_masks[idx]),  # attention mask 추가
                    torch.tensor(self.targets[0][idx]),
                    torch.tensor(self.targets[1][idx]),
                )
            else:
                return (
                    torch.tensor(self.inputs[idx]),
                    torch.tensor(self.segment_ids[idx]),
                    torch.tensor(self.attention_masks[idx]),
                    torch.tensor(self.targets[idx]),
                )

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        shuffle,
        train_path,
        dev_path,
        test_path,
        predict_path,
        multi_task,
    ):
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
        self.target_columns = ["label"]
        self.multi_task = multi_task
        if self.multi_task:
            self.target_columns2 = ["label", "binary-label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence_1", "sentence_2"]

    def tokenizing(self, dataframe):
        data = []
        segment_ids = []  # segment embeddings을 위한 리스트 추가
        attention_masks = []  # attention mask를 위한 리스트 추가

        for idx, item in tqdm(dataframe.iterrows(), desc="tokenizing", total=len(dataframe)):
            text1 = item[self.text_columns[0]]
            text2 = item[self.text_columns[1]]

            # 두 입력 문장을 토크나이징하고 segment embeddings 및 attention mask 추가
            # outputs 형태: [CLS] + 문장1 토큰들 + [SEP] + 문장2 토큰들 + [SEP]
            outputs = self.tokenizer(
                text1,
                text2,
                add_special_tokens=True,
                padding=False,  # 패딩을 tokenizer 단계에서는 수행하지 않음
                truncation=False,  # 잘리지 않도록 설정
            )

            input_ids = outputs["input_ids"]
            token_type_ids = outputs["token_type_ids"]  # segment embeddings
            attention_mask = outputs["attention_mask"]  # attention mask 추가

            data.append(input_ids)
            segment_ids.append(token_type_ids)
            attention_masks.append(attention_mask)  # attention mask 저장

        return data, segment_ids, attention_masks  # attention mask 반환하도록 수정

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
            if self.multi_task:
                targets = [data[col].values.tolist() for col in self.target_columns2]
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs, segment_ids, attention_masks = self.tokenizing(data)  # segment_ids, attention_masks 추가

        return inputs, segment_ids, attention_masks, targets  # segment_ids, attention_masks 반환

    def setup(self, stage="fit"):
        if stage == "fit":
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_segment_ids, train_attention_masks, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_segment_ids, val_attention_masks, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(
                train_inputs, train_segment_ids, train_attention_masks, train_targets, self.multi_task
            )
            self.val_dataset = Dataset(val_inputs, val_segment_ids, val_attention_masks, val_targets, self.multi_task)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_segment_ids, test_attention_masks, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(
                test_inputs, test_segment_ids, test_attention_masks, test_targets, self.multi_task
            )

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_segment_ids, predict_attention_masks, predict_targets = self.preprocessing(
                predict_data
            )
            self.predict_dataset = Dataset(
                predict_inputs, predict_segment_ids, predict_attention_masks, [], self.multi_task
            )

    def dynamic_padding_collate_fn(self, batch):
        """
        주어진 배치에서 동적 패딩을 적용하여 input_ids, segment_ids, attention_mask를 같은 길이로 맞추고 반환합니다.

        Parameters:
        batch (list of tuples): DataLoader에서 제공하는 배치 데이터.
                                각 요소는 (input_ids, segment_ids, attention_mask, target1, target2)와 같은 구조를 가질 수 있습니다.
                                multi_task가 False인 경우 (input_ids, segment_ids, attention_mask, target1)
                                추론 단계에서는 (input_ids, segment_ids, attention_mask)로 구성됩니다.

        Returns:
        tuple: 패딩된 입력 및 필요한 target을 포함한 튜플을 반환합니다.
            - multi_task=False일 경우: (padded_inputs, padded_segment_ids, padded_attention_masks, targets)
            - multi_task=True일 경우: (padded_inputs, padded_segment_ids, padded_attention_masks, targets1, targets2)
            - 추론 단계일 경우: (padded_inputs, padded_segment_ids, padded_attention_masks)

        Notes:
        - 각 문장을 해당 배치의 가장 긴 문장에 맞게 패딩합니다.
        - attention_mask와 segment_ids도 동일한 길이로 패딩됩니다.
        - self.tokenizer.pad_token_id를 사용하여 input_ids에 패딩을 적용합니다.

        Example:
            # DataLoader를 이용한 사용 예시
            from torch.utils.data import DataLoader
            dataset = Dataset(inputs, segment_ids, attention_masks, targets, multi_task=False)
            dataloader = DataLoader(dataset, batch_size=4, collate_fn=dynamic_padding_collate_fn)
            for batch in dataloader:
                padded_inputs, padded_segment_ids, padded_attention_masks, targets = batch
                print(padded_inputs.shape)  # torch.Size([batch_size, max_length])
                print(padded_segment_ids.shape)  # torch.Size([batch_size, max_length])
                print(padded_attention_masks.shape)  # torch.Size([batch_size, max_length])
                print(targets.shape)  # torch.Size([batch_size])
        """
        inputs = [item[0] for item in batch]
        segment_ids = [item[1] for item in batch]
        attention_masks = [item[2] for item in batch]  # attention mask 추가

        # 가장 긴 input의 길이로 padding
        max_length = max(len(input) for input in inputs)

        padded_inputs = torch.cat(
            [
                torch.cat([input, torch.full((max_length - len(input),), self.tokenizer.pad_token_id)])
                for input in inputs
            ]
        ).view(
            len(inputs), max_length
        )  # torch.cat의 결과는 1차원으로 나와서 2차원으로 배치 차원을 유지하기 위해 view를 사용

        padded_segment_ids = torch.cat(
            [
                torch.cat([segment_id, torch.full((max_length - len(segment_id),), -1, dtype=torch.long)])
                for segment_id in segment_ids
            ]
        ).view(len(segment_ids), max_length)

        padded_attention_masks = torch.cat(
            [torch.cat([mask, torch.zeros(max_length - len(mask), dtype=torch.long)]) for mask in attention_masks]
        ).view(len(attention_masks), max_length)

        if len(batch[0]) == 4:
            targets = torch.tensor([item[3] for item in batch])
            return padded_inputs, padded_segment_ids, padded_attention_masks, targets
        elif len(batch[0]) == 5:
            targets1 = torch.tensor([item[3] for item in batch])
            targets2 = torch.tensor([item[4] for item in batch])
            return padded_inputs, padded_segment_ids, padded_attention_masks, targets1, targets2

        return padded_inputs, padded_segment_ids, padded_attention_masks

    def train_dataloader(self):
        # return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.dynamic_padding_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=self.dynamic_padding_collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=self.dynamic_padding_collate_fn
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.batch_size, collate_fn=self.dynamic_padding_collate_fn
        )
