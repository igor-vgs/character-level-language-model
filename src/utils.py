import os
import random

import torch
import numpy as np


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(0)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_epoch(
    model,
    optimizer,
    loader,
    device,
    criterion,
    epoch_num,
    log_step=300,
    use_batches=-1,
):
    if use_batches == -1:
        use_batches = len(loader)
    train_loss = []
    tmp_loss = []
    for idx, batch in enumerate(loader):
        optimizer.zero_grad()
        target = batch['target'].to(device)
        del batch['target']
        for key in batch:
            batch[key] = batch[key].to(device)
        model_output = model(**batch)

        loss = criterion(model_output.view(-1, model_output.shape[-1]), target.view(-1))
        loss.backward()
        optimizer.step()
        tmp_loss.append(loss.cpu().detach().item())
        if (idx + 1) % log_step == 0:
            train_loss.append(sum(tmp_loss) / len(tmp_loss))
            tmp_loss = []
            print(f'[{epoch_num}] {idx}/{len(loader)}\ttrain loss: {train_loss[-1]}')
        
        if (idx + 1) % use_batches == 0:
            break
    return train_loss


def validate(model, loader, device, criterion, use_batches=-1):
    if use_batches == -1:
        use_batches = len(loader)
    valid_loss, valid_ppl = [], []
    for idx, batch in enumerate(loader):
        target = batch['target'].to(device)
        del batch['target']
        for key in batch:
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            model_output = model(**batch)
        loss = criterion(model_output.view(-1, model_output.shape[-1]), target.view(-1))
        valid_loss.append(loss.cpu().detach().item())
        if (idx + 1) % use_batches == 0:
            break
    loss_val = sum(valid_loss) / len(valid_loss)
    ppl_val = 2 ** loss_val
    print('########################################')
    print(f'Valid loss: {loss_val}\nValid PPL: {ppl_val}')
    print('########################################')
    return loss_val, ppl_val


def make_checkpoint(
    model,
    metric_score,
    checkpoint_dir='./checkpoints/',
    current_checkpoint=None,
    prefix='char_level_gru',
    metric_name='val_ppl'
):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    prev_metric_score = float('inf')
    if current_checkpoint is not None:
        metric_str = current_checkpoint.split(metric_name)[1]
        prev_metric_score = float(metric_str.replace('_', '').replace('.ckpt', ''))
    print(f'Current metric: {metric_score}\nPrev metric: {prev_metric_score}')
    new_ckpt_name = current_checkpoint
    if metric_score < prev_metric_score:
        print('Update checkpoint')
        if current_checkpoint is not None and os.path.exists(current_checkpoint):
            os.remove(current_checkpoint)
        new_ckpt_name = checkpoint_dir + prefix + '_' + metric_name + f'_{metric_score}.ckpt'
        torch.save(model.state_dict(), new_ckpt_name)
        print(f'Save {new_ckpt_name}')
    else:
        print('Dont update checkpoint')
    print('########################################')
    return new_ckpt_name
