from facenet_mini import base_network,Pairs,Triplet,Dataset
from argparse import ArgumentParser

parser = ArgumentParser("Train Custom Facenet")
parser.add_argument("--path",metavar='P',type=str,help="dataset path")


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = Dataset(path=args.path,n_faces=100)

    dataset.parse()