import random

import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import module.dataset_dual
from module.dataset import Dataloader
from module.layer_aggregated_similarity_model import LayerAggregatedSimilarityModel
from module.model import Model, MultitaksModel
from module.sbert_model import SBERTModel

# seed 고정
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)


if __name__ == "__main__":

    with open("config/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_name = config["model"]["model_name"]
    batch_size = config["train"]["batch_size"]
    max_epoch = config["train"]["max_epochs"]
    fine_tuning_epoch = config["train"]["fine_tuning_epoch"]
    shuffle = config["train"]["shuffle"]
    learning_rate = float(config["train"]["learning_rate"])
    train_path = config["data"]["train_path"]
    dev_path = config["data"]["dev_path"]
    test_path = config["data"]["test_path"]
    predict_path = config["data"]["predict_path"]
    loss_function = config["train"]["loss_function"]
    multi_task = config["train"]["multi_task"]
    siamese = config["model"]["siamese"]
    layer_aggregate = config["model"]["layer_aggregate"]

    save_path = f"./save_model/{model_name.replace('/','-')}_Batch-size:{batch_size}_Max-epochs:{max_epoch}_Multi-task:{multi_task}_lr:{learning_rate}_siamese:{siamese}/"
    config["data"]["model_path"] = save_path + "model.pt"

    early_stopping_callbacks = EarlyStopping(monitor="val_pearson", patience=7, mode="max")
    model_checkpoint_callbacks = ModelCheckpoint(
        monitor="val_pearson",
        dirpath=save_path,
        save_top_k=1,
        mode="max",
        filename="first_model",
    )

    wandblogger = WandbLogger(
        project="STSporj",
        name=f"after_final{model_name.replace('/','-')}real1alllearning//loss_func:{loss_function}//batch_size:{batch_size}//Multi_task:{multi_task}//Siamese:{siamese}//lr:{learning_rate}//CNN:weightslearning",
    )

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

    # <person> token 추가.
    person_token = ["<person>"]
    dataloader.token_add(person_token)

    if multi_task:
        weight = 0.7
        model = MultitaksModel(model_name, learning_rate, loss_function, weight)
        model.encoder.resize_token_embeddings(len(dataloader.tokenizer))

        # # Fine-tuning을 위한 설정
        # for param in model.encoder.parameters():
        #     param.requires_grad = False  # 전체 파라미터 freeze
        # model.similarity_head.requires_grad = True  # similarity head 업데이트 가능
        # model.classification_head.requires_grad = True  # classification head 업데이트 가능

        # # 해당 인코더 레이어 업데이트 가능 (예를 들어 마지막 2개 layer만 업데이트는 [-2:])
        # for layer in model.encoder.layer[-2:]:
        #     for param in layer.parameters():
        #         param.requires_grad = True  # 해당 레이어 업데이트 가능
    else:
        if siamese:
            model = SBERTModel(model_name, learning_rate)
        elif layer_aggregate:
            # model = torch.load("./save_model/2linearcnnklue-roberta-large_Batch-size:64_Max-epochs:20_Multi-task:False_lr:5e-07_siamese:False/model_CNN_weightslearning2cnnlinear5e-7.pt")
            model = LayerAggregatedSimilarityModel(model_name, learning_rate, loss_function)

        else:
            model = Model(model_name, learning_rate, loss_function)
        model.plm.resize_token_embeddings(len(dataloader.tokenizer))

        # # Fine-tuning을 위한 설정
        # for param in model.plm.parameters():
        #     param.requires_grad = False  # 전체 파라미터 freeze
        # model.plm.classifier.requires_grad = True  # 마지막 레이어 업데이트 가능

        # 해당 레이어 업데이트 가능 (예를 들어 마지막 2개 layer만 업데이트는 [-2:])
        # for layer in model.plm.encoder.layer[-2:]:
        #     for param in layer.parameters():
        #         param.requires_grad = True  # 해당 레이어 업데이트 가능

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

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, save_path + "model.pt")

    # # Pearson 상관계수에 맞게 마지막 레이어만 미세 조정
    # model.fine_tuning = True
    # trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=fine_tuning_epoch, log_every_n_steps=1, callbacks=[early_stopping_callbacks, model_checkpoint_callbacks], logger=wandblogger)  # fine-tuning을 위한 Trainer 생성
    # trainer.fit(model=model, datamodule=dataloader)  # fine-tuning 수행
    # trainer.test(model=model, datamodule=dataloader)  # fine-tuning 후 테스트

    wandb.finish()
    torch.save(model, save_path + "afterfianlmodel_ALL_weightslearning25e-6.pt")
