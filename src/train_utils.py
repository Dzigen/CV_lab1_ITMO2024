import torch
import transformers
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import gc
import json
from time import time
import os
import albumentations as A
import cv2
from dataclasses import dataclass

def param_count(model):
    return sum([p.numel() for name, p in model.named_parameters() if p.requires_grad])

def train(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    process = tqdm(loader)
    for batch in process:
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(batch['images'])
        loss = criterion(output, batch['labels'])

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})

    return losses

def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    predicted_labels = []
    target_labels = []

    process = tqdm(loader)
    for batch in process:

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        output = model(batch['images'])
        loss = criterion(output, batch['labels'])
        predicted = torch.argmax(output, dim=-1)

        losses.append(loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})

        #
        predicted_labels += predicted.detach().cpu()
        target_labels += batch['labels'].detach().cpu()
    
    f1_macro = f1_score(target_labels, predicted_labels, average='macro')
    acc_score = accuracy_score(target_labels, predicted_labels)

    return losses, {'f1_macro': f1_macro, 'accuracy': acc_score}

def run(config, model, train_loader, eval_loader):

    print("Model parameters count: ",param_count(model))

    print("Init folder to save")
    run_dir = f"{config.base_dir}/{config.run_name}"
    if os.path.isdir(run_dir):
        print("Error: Директория существует")
        return
    os.mkdir(run_dir)
    logs_file_path = f'{config.base_dir}/{config.run_name}/logs.txt'
    path_to_best_model_save = f"{config.base_dir}/{config.run_name}/bestmodel.pt"

    print("Saving used nn-arch")
    with open(f"{config.base_dir}/{config.run_name}/used_arch.txt", 'w', encoding='utf-8') as fd:
        fd.write(model.__str__())

    print("Saving used config")
    with open(f"{run_dir}/used_config.json", 'w', encoding='utf-8') as fd:
        json.dump(config.__dict__, indent=2, fp=fd)

    print("Init train objectives")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    ml_train = []
    ml_eval = []
    eval_scores = []
    best_score = 0

    print("===LEARNING START===")
    for i in range(config.epochs):

        print(f"Epoch {i+1} start:")
        train_s = time()
        train_losses = train(model, train_loader, 
                             optimizer, criterion, config.device)
        train_e = time()
        eval_losses, eval_metrics = evaluate(model, eval_loader, 
                                            criterion, config.device)
        eval_e = time()
        
        torch.cuda.empty_cache()
        gc.collect()

        #
        if best_score <= eval_metrics['f1_macro']:
            print("Update Best Model")
            if config.to_save:
                torch.save(model.state_dict(), path_to_best_model_save)
            best_score = eval_metrics['f1_macro']

        #
        ml_train.append(np.mean(train_losses))
        ml_eval.append(np.mean(eval_losses))
        eval_scores.append(eval_metrics)
        print(f"Epoch {i+1} results: tain_loss - {round(ml_train[-1], 5)} | eval_loss - {round(ml_eval[-1],5)}")
        print(eval_scores[-1])

        # Save train/eval info to logs folder
        epoch_log = {
            'epoch': i+1, 'train_loss': ml_train[-1],
            'eval_losss': ml_eval[-1], 'scores': eval_scores[-1],
            'train_time': round(train_e - train_s, 5), 'eval_time': round(eval_e - train_e, 5)
            }
        with open(logs_file_path,'a',encoding='utf-8') as logfd:
            logfd.write(str(epoch_log) + '\n')

    print("===LEARNING END===")
    print("Best score: ", best_score)

    return ml_train, ml_eval, best_score, eval_scores