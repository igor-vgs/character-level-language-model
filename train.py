import os
import argparse

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.dataset import WordsDataset
from src.tokenizer import Tokenizer
from src.model import CharLanguageModel
from src.utils import seed_all, train_epoch, validate, make_checkpoint


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train_path', type=str, default='./data/wikitext-2/wiki.train.tokens')
    arg_parser.add_argument('--valid_path', type=str, default='./data/wikitext-2/wiki.valid.tokens')
    arg_parser.add_argument('--max_length', type=int, default=20)
    arg_parser.add_argument('--batch_size', type=int, default=500)
    arg_parser.add_argument('--embedding_size', type=int, default=100)
    arg_parser.add_argument('--hidden_size', type=int, default=50)
    arg_parser.add_argument('--learning_rate', type=float, default=1e-3)
    arg_parser.add_argument('--train_epochs', type=int, default=20)
    arg_parser.add_argument('--log_step', type=int, default=1500)
    arg_parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    arg_parser.add_argument('--prefix', type=str, default='char_level_gru')
    arg_parser.add_argument('--metric_name', type=str, default='val_ppl')
    arg_parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    arg_parser.add_argument('--seed', type=int, default=42)
    args = arg_parser.parse_args()

    seed_all(args.seed)
    print('########################################')
    print('Load data')
    train_data = WordsDataset._read_data(args.train_path)
    valid_data = WordsDataset._read_data(args.valid_path)

    print(f'Train size: {len(train_data)}\nValid size: {len(valid_data)}')
    print('########################################')

    tokenizer = Tokenizer(train_data)

    print('########################################')
    print('Build datasets')
    train_dataset = WordsDataset(train_data, tokenizer, max_length=args.max_length)
    valid_dataset = WordsDataset(valid_data, tokenizer, max_length=args.max_length)

    print('########################################')
    print('Build dataloaders')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    VOCAB_SIZE = tokenizer.vocab_size

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f'Device: {device}')

    model = CharLanguageModel(
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size, 
        vocab_size=VOCAB_SIZE,
        pad_id=tokenizer.pad_id
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loss, valid_loss, valid_ppl = [], [], []
    best_checkpoint = None

    print(f'Train {args.train_epochs} epochs')
    for epoch in range(args.train_epochs):
        print(f'Start epoch {epoch}')
        model.train()
        epoch_train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            criterion=criterion,
            epoch_num=epoch,
            log_step=args.log_step,
            use_batches=-1
        )
        train_loss.extend(epoch_train_loss)
        model.eval()
        print('Start validation')
        val_loss, val_ppl = validate(
            model=model,
            loader=valid_loader,
            device=device,
            criterion=criterion,
            use_batches=-1
        )
        best_checkpoint = make_checkpoint(
            model,
            val_ppl,
            checkpoint_dir=args.checkpoint_dir,
            current_checkpoint=best_checkpoint,
            prefix=args.prefix,
            metric_name=args.metric_name
        )
        valid_loss.append(val_loss)
        valid_ppl.append(val_ppl)

    fig, axs = plt.subplots(3, figsize=(10, 10))

    axs[0].plot(train_loss, '-*')
    axs[0].grid()
    axs[0].set_ylabel('train loss')

    axs[1].plot(list(range(args.train_epochs)), valid_loss, '-*')
    axs[1].grid()
    axs[1].set_ylabel('valid loss')
    axs[1].set_xlabel('epochs')

    axs[2].plot(list(range(args.train_epochs)), valid_ppl, '-*')
    axs[2].grid()
    axs[2].set_xlabel('epochs')

    if not os.path.exists('./pics'):
        os.mkdir('./pics')
    best_checkpoint_ppl = best_checkpoint.split(args.metric_name)[1]
    best_checkpoint_ppl = best_checkpoint_ppl.replace('_', '').replace('.ckpt', '')
    best_checkpoint_ppl = float(best_checkpoint_ppl)
    fig.savefig(f'./pics/graphics_{args.train_epochs}_epochs_{best_checkpoint_ppl}_ppl.png')
