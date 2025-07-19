import argparse
import torchvision
from torchvision import transforms
from datasets import load_dataset
from dataset import HuggingFace
from torch.utils.data import DataLoader
import wandb
import optax
from model import VQGAN
from discriminator import Discriminator
from lpips import LPIPS


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)

    # Dataset
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)

    # VQGAN
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--num-embed', type=int, default=1024)
    parser.add_argument('--commitment-cost', type=float, default=0.25)
    parser.add_argument('--img-channel', type=int, default=3)
    parser.add_argument('--lr-rate', type=float, default=0.0001)
    parser.add_argument('--num-epochs', type=int, default=100)

    # Discriminator
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--disc_start', type=int, default=5000)

    # Wandb
    parser.add_argument('--project', type=str, default='VQ-GAN')
    parser.add_argument('--name', type=str, default='run_standard')

    # Save
    parser.add_argument('--checkpoint-path', type=str, required=True)

    return parser.parse_args(args)


def everything(args):
    args = parse_args(args)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),  # Normalize [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale [-1, 1]
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Convert [C, H, W] to [H, W, C]
        transforms.Lambda(lambda x: x.numpy()),
    ])

    train_dataset = HuggingFace(
        dataset=load_dataset("flwrlabs/celeba", split='train'),
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    vqgan = VQGAN(
        embedding_dim=args.embed_dim,
        num_embeddings=args.num_embed,
        commitment_cost=args.commitment_cost,
        output_channels=args.img_channel,
        channel_multipliers=[1, 1, 2, 2, 4],
    )

    discriminator = Discriminator(
        channel_multipliers=[1, 2, 4, 8],
        base_channels=args.base_channels,
    )

    lpips = LPIPS(channels=[64, 128, 256, 512, 512])

    vqgan_optimizer = optax.chain(
        optax.adam(
            learning_rate=args.lr_rate,
            b1=0.0,
            b2=0.99,
        ))

    disc_optimizer = optax.chain(optax.adam(
        learning_rate=args.lr_rate,
        b1=0.0,
        b2=0.99,
    ))

    epochs = args.num_epochs

    run = wandb.init(
        project=args.project,
        name=args.name,
        reinit=True,
        config=vars(args)
    )

    return {
        'seed': args.seed,
        'train_loader': train_loader,
        'vqgan': vqgan,
        'disc': discriminator,
        'lpips': lpips,
        'vqgan_optimizer': vqgan_optimizer,
        'disc_optimizer': disc_optimizer,
        'disc_start': args.disc_start,
        'epochs': epochs,
        'run': run,
        'checkpoint_path': args.checkpoint_path,
    }
