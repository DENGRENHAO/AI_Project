import os
import cv2
import numpy as np
from cairosvg import svg2png
from pathlib import Path
import shutil

def open_img(path, size):
    full_path = Path(path).resolve().absolute()
    file_name = os.path.split(full_path)[-1]
    name, ext = os.path.splitext(file_name)

    data = None
    if ext == '.png':
        data = cv2.imread(str(full_path))
        data = cv2.resize(data, (size, size))
        return data
    elif ext == '.svg':
        with open(full_path, 'r') as file:
            data = file.read()
        
        data = svg2png(
            bytestring=data,
            output_height=size,
            output_width=size
        )
        data = np.frombuffer(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data
    else:
        return []


def get_dir_content(path):
    res = {}
    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        if os.path.isfile(full_path):
            file_name = os.path.split(full_path)[-1]
            name, ext = os.path.splitext(file_name)
            res[name] = full_path
            
    return res


def create_dataset(in_dirs, out_path, out_img_size, paired=False):
    out_dirs = []
    if paired:
        assert len(in_dirs) == 2
        out_dirs = ['paired']
    else:
        for dir in in_dirs:
            name = os.path.split(dir)[-1]
            out_dirs.append(name)

    for i in range(len(out_dirs)):
        out_dirs[i] = os.path.join(out_path, out_dirs[i])

    if not paired:
        for i in range(len(in_dirs)):
            preprocess_input(in_dirs[i], out_dirs[i], out_img_size)
    else:
        theme_source = in_dirs[0]
        theme_target = in_dirs[1]
        for name in theme_target:
            if not name in theme_source:
                continue
            try:
                out_path = os.path.join(out_dirs[0], name + '.png')
                img1 = open_img(theme_target[name], out_img_size)
                img2 = open_img(theme_source[name], out_img_size)
                img = cv2.hconcat([img1, img2])
                if len(img):
                    cv2.imwrite(out_path, img)
            except:
                continue
    
    return out_dirs


def preprocess_input(input_dir, tmp_out, img_size):
    if os.path.exists(tmp_out):
        shutil.rmtree(tmp_out)
    os.makedirs(tmp_out)

    theme = get_dir_content(input_dir)
    for name in theme:
        file_path = theme[name]
        out_path = os.path.join(tmp_out, name + '.png')
        img = open_img(file_path, img_size)
        if len(img):
            cv2.imwrite(out_path, img)