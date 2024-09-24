import argparse

import pandas as pd

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import pytorch_lightning as pl
from module.dataset import Dataloader
## 맞춤법검사전처리.

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='monologg/koelectra-base-discriminator', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='./data/train_preprocessed_augmented.csv')
    parser.add_argument('--loss_function', default='L1')
    parser.add_argument('--dev_path', default='./data/val_preprocessed.csv')
    parser.add_argument('--test_path', default='./data/val_preprocessed.csv')
    parser.add_argument('--predict_path', default='/data/ephemeral/home/level1-semantictextsimilarity-nlp-015/data/test_preprocessed.csv')
    parser.add_argument('--multi_task', default=True)
    args = parser.parse_args(args=[])

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path,args.multi_task)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model_path = "/data/ephemeral/home/level1-semantictextsimilarity-nlp-015/save_model/klue-roberta-large_Batch-size:64_Max-epochs:20/model.pt"
    model = torch.load(model_path)
    
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('../sample_submission.csv')
    output['target'] = predictions
    output.to_csv('output_5.csv', index=False)
