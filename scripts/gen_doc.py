import os

import yaml

CONFIG = './config.yaml'

TEMPLATE_DIR = '../templates'
MARKDOWN_DIR = '../docs/code'

nav = []

config = {}

with open(CONFIG, 'r', encoding='utf-8') as f:
    config = yaml.load(f)

def make_dir(layer: int, dirname: str):
    
    name = dirname
    nav.append('  ' * layer + '- ' + config['title'][name] + ': ' + '\n')


def make_file(layer: int, raw_path: str, filename: str):
    name = filename.removesuffix('.cpp')
    
    nav.append('  ' * layer + '- ' + config['title'][name] + ': ' + 'code/' + name + '.md' + '\n')
    out_path = MARKDOWN_DIR + '/' + name + '.md'

    lines = []

    with open(raw_path, 'r', encoding='utf-8') as raw:
        lines = raw.readlines()
    
    with open(out_path, 'w', encoding='utf-8') as out:
        p = -1
        if len(lines) > 1 and lines[0] == '/**\n':
            p = lines.index('**/\n')

            out.writelines(lines[1:p])
            out.write('\n')
        
        out.write('```cpp\n')
        out.writelines(lines[p + 1:])
        out.write('\n```\n')

nav.append('nav:\n')

for layer1 in os.scandir(TEMPLATE_DIR):
    if layer1.is_dir():
        make_dir(1, layer1.name)
        for layer2 in os.scandir(layer1.path):
            if layer2.is_dir():
                make_dir(2, layer2.name)
                for layer3 in os.scandir(layer2.path):
                    make_file(3, layer3.path, layer3.name)
            elif layer2.is_file():
                make_file(2, layer2.path, layer2.name)
    elif layer1.is_file():
            make_file(1, layer1.path, layer1.name)

with open('nav.txt', 'w', encoding='utf-8') as f:
    f.writelines(nav)