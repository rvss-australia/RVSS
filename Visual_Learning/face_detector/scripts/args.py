import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_classes", metavar='', type=int, default=2)
parser.add_argument("--lr", metavar='', type=float, default=1.0e-3)
parser.add_argument("--epochs", metavar='', type=int, default=10)
parser.add_argument("--batch_size", metavar='', type=int, default=16)
parser.add_argument("--weight_decay", metavar='', type=float, default=1.0e-4)
parser.add_argument("--scheduler_step", metavar='', type=int, default=5,
        help='the learning rate is reduced to <?>')
parser.add_argument("--scheduler_gamma", metavar='', type=float, default=0.5,
        help='the learning rate is reduced to <?> every <?> epochs')
# Trainer Configuration
parser.add_argument("--model_dir", metavar='', type=str, default="",
                    help="folder to save the experiment")
parser.add_argument("--model", metavar='', type=str, default="res18_baseline",
                    help="specify network architecture here")
parser.add_argument("--load_best", metavar='', type=int, default=0, 
    help=' load the best checkpoint')
parser.add_argument("--log_freq", metavar='', type=int, default=2,
                    help=" record a training log every <?> mini-batches")
parser.add_argument('--dataset_dir', metavar='', type=str, default="dataset/")

args, _ = parser.parse_known_args()

try: 
    args.model_dir=os.environ['SM_MODEL_DIR']
    args.dataset_dir = os.environ['SM_CHANNEL_TRAIN']
except:
    pass

