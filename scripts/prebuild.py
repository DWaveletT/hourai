import os

import yaml

CONFIG = './config.yaml'

TEMPLATE_DIR = '../templates'
MARKDOWN_DIR = '../docs/code'

PRINT_DIR = '../docs/print'

nav = []
all = []
rec = {}

config = {}

with open(CONFIG, 'r', encoding='utf-8') as f:
    config = yaml.load(f)

def make_dir(layer: int, dirname: str):
    name = dirname
    print('make dir:  ', name)

    nav.append('  ' * layer + '- {}:\n'.format(
        config['title'][name]
    ))
    all.append('#' * layer + ' {}\n\n'.format(config['title'][name]))

def make_file(layer: int, raw_path: str, filename: str):
    name = filename.removesuffix('.cpp')
    print('make file: ', name)

    out_path = '{}/{}.md'.format(MARKDOWN_DIR, name)

    lines = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    nav.append('  ' * layer + '- {}: code/{}.md\n'.format(
        config['title'][name], name
    ))
    out = []
    all.append('#' * layer + ' {}\n\n'.format(config['title'][name]))

    p = -1
    if len(lines) > 1 and lines[0] == '/**\n':
        p = lines.index('**/\n')
        for line in lines[1:p]:
            if line.startswith('#'):
                all.append('#' * (layer - 1) + '{}'.format(line))
            else:
                all.append(line)
        
        all.append('\n')
        out.extend(lines[1:p]), out.append('\n')
    
    out.append('```cpp\n')
    for line in lines[p + 1:]:
        if line.split():
            out.extend(line)
    out.append('\n```\n')
    all.append('```cpp\n')
    for line in lines[p + 1:]:
        if line.split():
            all.extend(line)
    all.append('\n```\n')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(out)
    rec['{}.md'.format(name)] = True

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

for file in os.scandir(MARKDOWN_DIR):
    if not file.name in rec.keys():
        print('unknown file: {}'.format(file.name))
        os.remove(file.path)

with open('nav.txt', 'w', encoding='utf-8') as f:
    f.writelines(nav)

with open('{}/full-version.md'.format(PRINT_DIR), 'w', encoding='utf-8') as f:
    f.writelines(all)
