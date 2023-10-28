import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf",
    help="path to json config file for hyperparameters",
    type=str,
    default="config/config.json",
)

parser.add_argument(
    "--epochs",
    help="training epochs",
    type=int,
    default=100,
)

parser.add_argument(
    "--lr",
    help="learning rate",
    type=float,
    default=0.00005,
)

parser.add_argument(
    "--batch_size",
    help="batch size",
    type=int,
    default=256,
)

parser.add_argument(
    "--schedule",
    help="update scheduler",
    type=str,
    default="linear", #exp step cos
)

parser.add_argument(
    "--log_dir",
    help="log directory",
    type=str,
    default="experiments",
)

parser.add_argument(
    "--tag",
    help="log tag",
    type=str,
    default="VIFGNN",
)





args = parser.parse_args()
