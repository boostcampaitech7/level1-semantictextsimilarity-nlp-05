import random

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from module.dataset import Dataloader
from module.model import Model, MultitaksModel

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


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

    save_path = f"./save_model/{model_name.replace('/','-')}_Batch-size:{batch_size}_Max-epochs:{max_epoch}/"
    config["data"]["model_path"] = save_path + "model.pt"

    early_stopping_callbacks = EarlyStopping(
        monitor="val_pearson", patience=7, mode="max"
    )
    model_checkpoint_callbacks = ModelCheckpoint(
        monitor="val_pearson",
        dirpath=save_path,
        save_top_k=1,
        mode="max",
        filename="first_model",
    )

    wandblogger = WandbLogger(
        project="STSporj",
        name=f"{model_name.replace('/','-')}//loss_func:{loss_function}//batch_size:{batch_size}//Multi_task:{multi_task}",
    )

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

    # <person> token 추가.
    person_token = ["<person>"]
    dataloader.token_add(person_token)

    if multi_task:
        weight = 0.7
        model = MultitaksModel(model_name, learning_rate, loss_function, weight)
        model.encoder.resize_token_embeddings(len(dataloader.tokenizer))
    else:
        model = Model(model_name, learning_rate, loss_function)
        model.plm.resize_token_embeddings(len(dataloader.tokenizer))

    with open("config/config.yaml", "w") as f:
        yaml.dump(config, f)

    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epoch,
        log_every_n_steps=1,
        callbacks=[early_stopping_callbacks, model_checkpoint_callbacks],
        logger=wandblogger,
    )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    wandb.finish()
    # 학습이 완료된 모델을 저장합니다.
    # f'{model_name}_{data_version}_{Loss function}.pt'
    #
    torch.save(model, save_path + "model.pt")
