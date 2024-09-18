import argparse
import random

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from module.model import Model
from module.dataset import Dataloader
import wandb
# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)



if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=25, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--train_path', default='../train_v2.csv')
    parser.add_argument('--dev_path', default='../dev.csv')
    parser.add_argument('--test_path', default='../dev.csv')
    parser.add_argument('--predict_path', default='../test.csv')
    args = parser.parse_args(args=[])


    early_stopping_callbacks = EarlyStopping(
        monitor = 'val_loss',
        patience= 3,
        mode = 'min'
    )
    model_checkpoint_callbacks = ModelCheckpoint(
        monitor = 'val_loss',
        dirpath='', ## 수정필요
        save_top_k=1,
        mode = 'min'
    )

    ##작성필요
    wandblogger = WandbLogger()

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = Model(args.model_name, args.learning_rate)

    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=args.max_epoch, 
        log_every_n_steps=1,
        callbacks= [early_stopping_callbacks,model_checkpoint_callbacks],
        logger = wandblogger
    )
    
    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    wandb.finish()
    # 학습이 완료된 모델을 저장합니다.
    # f'{model_name}_{data_version}_{Loss function}.pt'
    #
    torch.save(model, 'robertaL_v2_mseloss.pt')
