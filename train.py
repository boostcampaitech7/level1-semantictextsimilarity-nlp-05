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
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=5e-6, type=float)
    parser.add_argument('--train_path', default='./data/train_preprocessed_augmented.csv')
    parser.add_argument('--loss_function', default='L1')
    parser.add_argument('--dev_path', default='./data/val_preprocessed.csv')
    parser.add_argument('--test_path', default='./data/val_preprocessed.csv')
    parser.add_argument('--predict_path', default='../test.csv')
    args = parser.parse_args(args=[])

    save_path = f"save_model/{args.model_name.replace('/','-')}_Batch-size:{args.batch_size}_Max-epochs:{args.max_epoch}/"

    early_stopping_callbacks = EarlyStopping(
        monitor = 'val_pearson',
        patience= 10,
        mode = 'max'
    )
    model_checkpoint_callbacks = ModelCheckpoint(
        monitor = 'val_pearson',
        dirpath=save_path, ## 수정필요
        save_top_k=1,
        mode = 'max',
        filename="first_model"
    )

    ##작성필요
    wandblogger = WandbLogger(project="STSporj",
                              name = f"{args.model_name.replace('/','-')}//loss_func:{args.loss_function}//batch_size:{args.batch_size}")

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = Model(args.model_name, args.learning_rate,args.loss_function)
    
    #<person> token 추가.
    person_token = ['<person>']
    dataloader.token_add(person_token)
    
    model.plm.resize_token_embeddings(len(dataloader.tokenizer))
    
    
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
    torch.save(model, save_path +'model.pt')
    