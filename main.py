import os
import shutil
import argparse
from preprocess import create_dataset, preprocess_input
from pix2pix import pix2pix

def parse_argument():
    parser = argparse.ArgumentParser(description='AI Final Project')
    parser.add_argument('-m','--mode', type=str,default='train',
                    help='train or test')
    
    parser.add_argument('-s','--source', type=str, default=None,
                    help='Path to a folder containing the source icons')
    parser.add_argument('-t','--target', type=str, default=None,
                    help='Path to a folder containing the target icons')
    parser.add_argument('-o','--out', type=str, default='out',
                    help='Path to the output folder, in trainning mode, it will be the output weight folder; in testing mode, it will be the output image folder')
    parser.add_argument('-w','--weight', type=str, default=None,
                    help='Path to the output folder')
    parser.add_argument('-M','--model', type=str, default='cyclegan',
                    help='CycleGAN or pix2pix or NST')
    parser.add_argument('--tmp', type=str, default='tmp',
                    help='Temporary folder ')
    
    return parser.parse_args()


def train(source, target, tmp, output,  model='cyclegan'):
    if not source or not target:
        print('Please provide source and target')
    
    if model.lower() == 'cyclegan':
        dirs = create_dataset([source, target], tmp, 64)
    elif model.lower() == 'pix2pix':
        dirs = create_dataset([source, target], tmp, 256, paired=True)
        pix2pix.train(dirs[0], target, output)
    else:
        print('Unsupported model:', model)
        return False
    
    return True


def test(weight, input, out, tmp, model='cyclegan'):
    in_path = os.path.join(tmp, 'test')
    if os.path.exists(in_path):
        shutil.rmtree(in_path)
    os.makedirs(in_path)

    if model.lower() == 'cyclegan':
        pass
    elif model.lower() == 'pix2pix':
        preprocess_input(input, in_path, 256)
        pix2pix.test(weight, in_path, out)
    else:
        print('Unsupported model:', model)
        return False
    
    return True


if __name__ == '__main__':
    args = parse_argument()
    if args.mode == 'train':
        train(args.source, args.target, args.tmp, args.out, args.model)
    elif args.mode == 'test':
        test(args.weight, args.out, args.model)
    else:
        print('Unknow mode: ' + args.mode)
