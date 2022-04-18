import os
import argparse
import cv2
from cairosvg import svg2png
from pathlib import Path

def find_icon_directorys():
    res = []

    data_dirs = os.environ['XDG_DATA_DIRS'].split(':')
    data_dirs += [ os.path.join(os.environ['HOME'], '.local', 'share') ]
    for data_dir in data_dirs:
        icons_path = os.path.join(data_dir, 'icons')
        if os.path.isdir(icons_path):
            for f in os.listdir(icons_path):
                file_path = os.path.join(icons_path, f)
                if os.path.isdir( file_path ):
                    res.append(file_path)
    
    return res


def get_icon_theme_dirs():
    icon_dirs = find_icon_directorys()
    res = {}
    for dir in icon_dirs:
        theme_name = os.path.split(dir)[-1]
        if not theme_name in res:
            res[theme_name] = []
        res[theme_name].append(dir)
    return res


def extract_theme_data(theme_paths, out, out_size):
    if not os.path.exists(out):
            os.makedirs(out)

    icon_dirs = []
    for path in theme_paths:
        pos_cat_dirs = [
            'apps',
            'devices',
            'mimetypes',
            'places',
            'preferences'
        ]
        pos_size_dirs =[
            str(out_size) + 'x' + str(out_size),
            str(out_size),
            'scalable'
        ]

        for cat_dir in pos_cat_dirs:
            for size_dir in pos_size_dirs:
                cur = os.path.join(path, cat_dir, size_dir)
                if os.path.isdir(cur):
                    icon_dirs.append(cur)
                cur = os.path.join(path, size_dir, cat_dir)
                if os.path.isdir(cur):
                    icon_dirs.append(cur)

    done = set()
    for dir in icon_dirs:
        for f in os.listdir(dir):
            full_path = os.path.join(dir, f)
            if os.path.islink(full_path):
                full_path = Path(full_path).resolve().absolute()
                f = os.path.split(full_path)[-1]

            if not os.path.isfile(full_path):
                continue
            name, ext = os.path.splitext(f)
            if name in done:
                continue
            
            data = None
            out_path = os.path.join(out, name + '.png')
            try:
                if ext == '.png':
                    data = cv2.imread(str(full_path))
                    data = cv2.resize(data, (out_size, out_size))
                    cv2.imwrite(out_path, data)
                    done.add(name)
                elif ext == '.svg':
                    with open(full_path, 'r') as file:
                        data = file.read()
                    
                    svg2png(
                        bytestring=data,
                        write_to=out_path,
                        output_height=out_size,
                        output_width=out_size
                    )
                    done.add(name)
            except:
                continue


def parse_cmd():
    parser = argparse.ArgumentParser(description='Intro_AI Project - Dataset Builder')
    parser.add_argument('-o','--out', type=str, default='dataset',
                    help='Output folder')
    parser.add_argument('-s','--size', type=int, default=128,
                    help='Size of the output image (size x size)')
    parser.add_argument('-B', '--build-dataset', action='store_true',
                    help='Build the dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cmd()
    if args.build_dataset:
        if not os.path.exists(args.out):
            os.makedirs(args.out)
        theme_dirs = get_icon_theme_dirs()
        for theme in theme_dirs.keys():
            extract_theme_data(theme_dirs[theme], os.path.join(args.out, theme), args.size)