from facenet_mini import base_network,Pairs,Triplet,Dataset
from argparse import ArgumentParser

parser = ArgumentParser("Train Custom Facenet")
parser.add_argument("--path",metavar='P',type=str,help="Dataset Path",required=True)
parser.add_argument("--n_faces",metavar='N',type=int,help="Number Of Faces",required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = Dataset(path=args.path,n_faces=args.n_faces)
    dataset.parse()

    pairs = Pairs()