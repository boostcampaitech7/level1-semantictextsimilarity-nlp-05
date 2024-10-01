import os

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml

import module.dataset_dual
from module.dataset import Dataloader
from module.model import Model, MultitaksModel
from module.sbert_model import SBERTModel

if __name__ == "__main__":

    with open("config/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_name = config["model"]["model_name"]
    batch_size = config["train"]["batch_size"]
    max_epoch = config["train"]["max_epochs"]
    shuffle = config["train"]["shuffle"]
    learning_rate = float(config["train"]["learning_rate"])
    train_path = config["data"]["train_path"]
    dev_path = config["data"]["dev_path"]
    test_path = config["data"]["test_path"]
    predict_path = config["data"]["predict_path"]
    submission_path = config["data"]["submission_path"]
    loss_function = config["train"]["loss_function"]
    multi_task = config["train"]["multi_task"]
    siamese = config["model"]["siamese"]

    if siamese:
        dataloader = module.dataset_dual.Dataloader(
            model_name,
            batch_size,
            shuffle,
            train_path,
            dev_path,
            test_path,
            predict_path,
        )
    else:
        # dataloader와 model을 생성합니다.
        dataloader = Dataloader(
            model_name,
            batch_size,
            shuffle,
            train_path,
            dev_path,
            test_path,
            predict_path,
            multi_task,
        )

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epoch, log_every_n_steps=1)

    def model_load(
        model_save_path, model_name, multi_task, siamese
    ):  # 현재 ckpt 파일을 불러오면 predictions 수치가 모두 5.0을 기록하는 오류가 있어서 .pt 파일을 사용해야 합니다
        """
        마지막 저장 모델을 불러오거나 peason 상관계수 최고일 때의 모델을 불러오는 것을 선택하는 함수
        model_save_path: .pt or .ckpt의 저장 경로
        model_name: 사용한 모델 이름
        multi_task: multi_task 사용 여부 True or False
        siamese: 문장1과 문장2를 분리한 입력으로 모델을 사용했는지 여부 True or False
        """
        if os.path.splitext(model_save_path)[-1] == ".pt":
            model = torch.load(model_save_path)
            return model
        elif os.path.splitext(model_save_path)[-1] == ".ckpt":
            # ckpt는 모델이 있어야 불러올 수 있으므로 사전 학습된 모델 선언
            if siamese:
                model = SBERTModel(model_name, learning_rate)
            elif multi_task:
                model = MultitaksModel(model_name, learning_rate, loss_function, weight=0.7)
            else:
                model = Model(model_name, learning_rate)
            checkpoint = torch.load(model_save_path, map_location=torch.device("cuda"))
            state_dict = checkpoint["state_dict"]  # ckpt 파일에서 모델 가중치(state_dict) 불러오기

            # 현재 모델의 키와 ckpt 파일의 키가 정확히 일치하는지 확인
            model_keys = model.state_dict().keys()
            ckpt_keys = state_dict.keys()

            # 필요한 키 변환 수행 (필요할 경우에만)
            # 예: Hugging Face Transformers 모델에 대한 불필요한 접두사 제거
            updated_state_dict = {}
            for key in ckpt_keys:
                new_key = key.replace("plm.", "")  # "plm." 접두사 제거
                updated_state_dict[new_key] = state_dict[key]

            # 불러온 가중치로 모델 업데이트
            model.load_state_dict(updated_state_dict, strict=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            return model

    model = model_load(
        model_save_path="./save_model/klue-roberta-large_Batch-size:64_Max-epochs:20_Multi-task:False_lr:5e-07_siamese:False/model_ALL_weightslearning25e-6.pt",
        model_name=model_name,
        multi_task=multi_task,
        siamese=siamese,
    )

    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))
    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    if "test" in predict_path:
        output = pd.read_csv("./data/sample_submission.csv")
        output["target"] = predictions
        output.to_csv("output.csv", index=False)
    else:
        # validation 데이터에 대해 정답 label과 예측 label의 비교를 하기 위한 csv 추출
        data_with_prediction = pd.read_csv(predict_path)
        data_with_prediction.insert(
            len(data_with_prediction.columns) - 1, f"prediction_{model_name}", predictions
        )  # 정답 label 비교를 쉽게 하기 위해 바로 옆에 예측한 label 배치
        output_file_name = "val_outputs.csv"
        print(output_file_name)
        # 파일을 저장할 디렉토리를 지정합니다.
        output_directory = "./data"
        os.makedirs(output_directory, exist_ok=True)
        output_file_path = os.path.join(output_directory, output_file_name)
        data_with_prediction.to_csv(output_file_path, index=False)
