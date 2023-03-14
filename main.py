import argparse
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from dataset import TwitterDataset
from model import UniTransformer
from tqdm import tqdm
from evaluation import evaluate


def parse_args():
    parser =argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default="Twitter2015")
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--seed",type=int,default=3407)
    parser.add_argument("--first_step_lr",type=float,default=5e-5)
    parser.add_argument("--first_step_weight_decay",type=float,default=1e-2)
    parser.add_argument("--max_len",type=int,default=197)
    parser.add_argument("--num_warmup_steps",type=int,default=10)
    parser.add_argument("--output_dir",type=str,default="output")
    parser.add_argument("--input_dim",type=int,default=768)
    parser.add_argument("--hidden_dim",type=int,default=768)
    parser.add_argument("--output_dim",type=int,default=768)
    parser.add_argument("--num_labels",type=int,default=5)
    parser.add_argument("--num_heads",type=int,default=8)
    parser.add_argument("--num_layers",type=int,default=8)
    parser.add_argument("--dropout_rate",type=float,default=0.1)
    parser.add_argument("--text_pretrained_model",type=str,default="roberta-base")
    parser.add_argument("--image_pretrained_model",type=str,default="google/vit-base-patch16-224")
    parser.add_argument("--path",type=str,default="C:/uni_transformer/UniTransformer/MABSA_datasets")
    parser.add_argument("--max_grad_norm",type=float,default=1.0)
    parser.add_argument("--second_step_lr",type=float,default=5e-5)
    parser.add_argument("--second_step_weight_decay",type=float,default=1e-2)
    parser.add_argument("--gpu",type=str,default="0")
    parser.add_argument("--pad_id",type=int,default=0)
    parser.add_argument("--patch_len",type=int,default=197)


    args = parser.parse_args()
    return args


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_2015_dataset = TwitterDataset(args,train=True)
    train_2015_dataloader = DataLoader(train_2015_dataset,batch_size=16,collate_fn=train_2015_dataset.collate_fn,shuffle=True)

    test_2015_dataset = TwitterDataset(args,train=False)
    test_2015_dataloader = DataLoader(test_2015_dataset, batch_size=1, collate_fn=test_2015_dataset.collate_fn,shuffle=True)

    model = UniTransformer(args).to(device)

    epochs = args.epochs
    max_grad_norm = args.max_grad_norm
    total_steps = len(train_2015_dataset) * epochs

    first_step_lr = args.first_step_lr
    first_step_weight_decay = args.first_step_weight_decay

    first_step_optimizer = torch.optim.AdamW(model.parameters(), lr=first_step_lr,weight_decay=first_step_weight_decay)
    first_step_scheduler = get_linear_schedule_with_warmup(first_step_optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)


    for epoch in tqdm(range(epochs)):
        losses = 0
        cnt = 0
        for input_ids, labels, images, attention_mask in train_2015_dataloader:
            cnt += 1
            model.train()

            input_ids, labels, images, attention_mask = input_ids.to(device), labels.to(device), images.to(device), attention_mask.to(device)
            loss = model(input_ids,images,attention_mask,labels)

            first_step_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=max_grad_norm)
            first_step_optimizer.step()
            first_step_scheduler.step()
            losses += loss.item()
        print("first_step_loss = {}".format(losses / cnt))


    model.two_stage = True
    second_step_lr = args.second_step_lr
    second_step_weight_decay = args.second_step_weight_decay

    second_step_optimizer = torch.optim.AdamW(model.parameters(), lr=second_step_lr,weight_decay=second_step_weight_decay)
    second_step_scheduler = get_linear_schedule_with_warmup(second_step_optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)


    for epoch in tqdm(range(epochs)):
        losses = 0
        cnt = 0
        for input_ids, labels, images, attention_mask in train_2015_dataloader:
            model.train()

            input_ids, labels, images, attention_mask = input_ids.to(device), labels.to(device), images.to(
                device), attention_mask.to(device)

            crf_loss = model(input_ids,images,attention_mask,labels)


            second_step_optimizer.zero_grad()
            crf_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            second_step_optimizer.step()
            second_step_scheduler.step()
            losses += crf_loss.item()

        if (epoch + 1) % 10 == 0:
            f1 = evaluate(test_2015_dataloader,model)
            if f1 > 0.7:
                torch.save(model.state_dict(), "best_model.pt")

        print("second_step_loss = {}".format(losses / cnt))



if __name__ == "__main__":
    args = parse_args()
    print("args",args)
    torch.manual_seed(args.seed)
    train(args)
