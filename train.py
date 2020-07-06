from argparse import ArgumentParser
from facenet_mini import builder,base_network,Pairs,Dataset
from facenet_mini.net import Triplet, Adagrad, StopTraining
from os import path as pathlib

parser = ArgumentParser("Train Custom Facenet")
parser.add_argument("--path",metavar='P',type=str,help="Dataset Path",required=True)
parser.add_argument("--n_faces",metavar='N',type=int,help="Number Of Faces",required=True)

parser.add_argument("--d",metavar='D',type=int,help="Vector dimentions",required=False,default=64)
parser.add_argument("--m",metavar='M',type=int,help="Multiplication factor",required=False,default=128)

parser.add_argument("--output",metavar='O',type=str,help="Output Dir",required=False,default="./facenet")
parser.add_argument("--margin",metavar='MR',type=int,help="Margin",required=False,default=4)
parser.add_argument("--epochs",metavar='E',type=int,help="Number of epochs",required=False,default=1000)
parser.add_argument("--batch_size",metavar='B',type=int,help="Batch size (while using batch pairs)",required=False,default=16)

if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path
    n_faces = args.n_faces
    vector_length = args.d
    multiplication_factor = args.m
    margin = args.margin
    epochs = args.epochs
    output = args.output
    
    assert pathlib.isdir(output),f"Directory {output} does not exists !"
    output = pathlib.abspath(output)

    print (f"""
[*] Traning Path          : {path}
[*] Number Of Faces       : {n_faces}
[*] Vector Length         : {vector_length}
[*] Multiplication Factor : {multiplication_factor}
[*] Margin                : {margin}
[*] Epochs                : {epochs}
[*] Output Path           : {output}
    """)
    
    dataset = Dataset(path=path,n_faces=n_faces)
    dataset.parse()
    model = base_network(vector_length=vector_length,multiplication_factor=multiplication_factor)
    train = builder(model=model)
    pairs = Pairs(model,dataset,)
    
    loss = Triplet(margin=margin,vector_length=vector_length)
    opt = Adagrad(0.0001)
    eh = StopTraining()
    
    f = pairs.flow(epochs=epochs)
    train.compile(optimizer=opt,loss=loss)
    train.fit_generator(f,epochs=epochs,steps_per_epoch=len(dataset.x),callbacks=[eh])
    
    open(f"{output}/facenet.json","w+").write(model.to_json())
    model.save_weights(f"{output}/facenet")