import pandas as pd
import pytorch_lightning as pl
import torch
import yaml

from module.dataset import Dataloader

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
    loss_function = config["train"]["loss_function"]
    multi_task = config["train"]["multi_task"]

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

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model_path = config["data"]["model_path"]
    model = torch.load(model_path)

    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv("../sample_submission.csv")
    output["target"] = predictions
    output.to_csv("output.csv", index=False)
