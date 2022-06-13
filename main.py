import os
import shutil
import argparse
from preprocess import create_dataset, preprocess_input
from pix2pix import pix2pix
from CycleGAN import CycleGAN


def parse_argument():
    parser = argparse.ArgumentParser(description='AI Final Project')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='train or test')

    parser.add_argument('-s', '--source', type=str, default='',
                        help='Path to a folder containing the source icons')
    parser.add_argument('-t', '--target', type=str, default='',
                        help='Path to a folder containing the target icons')
    parser.add_argument('-o', '--out', type=str, default='out',
                        help='Path to the output folder, in training mode, it will be the output weight folder; in testing mode, it will be the output image folder')
    parser.add_argument('-w', '--weight', type=str, default='',
                        help='Path to the folder containing saved weights, normally the trainning output folder')
    parser.add_argument('-M', '--model', type=str, default='cyclegan',
                        help='CycleGAN or pix2pix or NST')
    parser.add_argument('--tmp', type=str, default='tmp',
                        help='Temporary folder ')

    return parser.parse_args()


def train(source, target, tmp, output, model='cyclegan'):
    if not source or not target:
        print('Please provide source and target')
        return False

    if model.lower() == 'cyclegan':
        dirs = create_dataset([source, target], tmp, 64)
        content_img_path = os.path.join(dirs[0], '*.png')
        style_img_path = os.path.join(dirs[1], '*.png')
        checkpoint_filepath = os.path.join(output, "{epoch}_epoch/model_checkpoints/cyclegan_checkpoints.{epoch}")
        CycleGAN.train(content_img_path, style_img_path, checkpoint_filepath, output)
    elif model.lower() == 'pix2pix':
        dirs = create_dataset([source, target], tmp, 256, paired=True)
        pix2pix.train(dirs[0], output)
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
        preprocess_input(input, in_path, 64)
        weight_file_path = os.path.join(weight, "40_epoch/model_checkpoints/cyclegan_checkpoints.40")
        input_path = os.path.join(in_path, "*.png")
        CycleGAN.test(input_path, weight_file_path, out)
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
        test(args.weight, args.source, args.out, args.tmp, args.model)
    else:
        print('Unknown mode: ' + args.mode)