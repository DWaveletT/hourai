#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import pymdownx.slugs
import pymdownx.arithmatex
from pathlib import Path
from typing import List, Dict, Any
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

STATIC_DIR = ROOT_DIR / 'static'
DOCS_DIR = ROOT_DIR / 'docs'

CONFIG_PATH = ROOT_DIR / 'config.yml'
TEMPLATE_DIR = ROOT_DIR / 'templates'
WEBSITE_WF_DIR = ROOT_DIR / 'docs' / 'wf'
WEBSITE_DIR = ROOT_DIR / 'docs'
OFFLINE_DIR = ROOT_DIR / 'docs' / 'offline'

MKDOCS_HEADER = ROOT_DIR / 'mkdocs.header.yml'
MKDOCS_OUTPUT = ROOT_DIR / 'mkdocs.yml'

class MkDocsBuilder:
    def __init__(self, config_path: Path, template_dir: Path, website_dir: Path, website_wf_dir: Path, offline_dir: Path,
                 mkdocs_header: Path, mkdocs_output: Path):
        self.config = self.load_config(config_path)
        self.template_dir = template_dir
        self.website_dir = website_dir
        self.website_wf_dir = website_wf_dir
        self.offline_dir = offline_dir
        self.mkdocs_header = mkdocs_header
        self.mkdocs_output = mkdocs_output

        self.nav_structure: List[Any] = []
        self.full_doc_lines: List[str] = []
        self.wf_doc_lines: List[str] = []
        self.wf_files_set = set(self.config.get('wf_files', []))

        self.generated_files: Dict[str, bool] = {}

    @staticmethod
    def load_config(path: Path) -> Dict:
        with path.open('r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.FullLoader) or {}

    def add_dir_to_nav(self, layer: int, dirname: str, parent_nav: List):
        title = self.config['title'].get(dirname, dirname)
        new_nav = {title: []}
        parent_nav.append(new_nav)
        self.full_doc_lines.append(f"{'#' * layer} {title}\n\n")
        self.  wf_doc_lines.append(f"{'#' * layer} {title}\n\n")
        print(f"[DIR] {dirname}")
        return new_nav[title]

    def add_file(self, layer: int, file_path: Path, parent_nav: List):
        name = file_path.stem
        title = self.config['title'].get(name, name)
        print(f"[FILE] {name}")

        is_wf = name in self.wf_files_set

        # nav 添加文件
        parent_nav.append({title: f"{name}.md"})
        self.full_doc_lines.append(f"{'#' * layer} {title}\n\n")
        if is_wf:
            self.wf_doc_lines.append(f"{'#' * layer} {title}\n\n")
        
        out_path = self.website_dir / f"{name}.md"
        lines = file_path.read_text(encoding='utf-8').splitlines(keepends=True)

        out_lines, code_lines = self.extract_content(lines, layer, is_wf)
        self.write_markdown(out_path, out_lines)

        self.generated_files[f"{name}.md"] = True

    def extract_content(self, lines: List[str], layer: int, wf: bool):
        """Extract comment and code block"""
        out_lines = []
        doc_lines = []
        p = -1
        if len(lines) > 1 and lines[0] == '/**\n':
            try:
                p = lines.index('**/\n')
            except ValueError:
                p = -1

            if p > 0:
                for line in lines[1:p]:
                    if line.startswith('#'):
                        doc_lines.append('#' * (layer - 1) + line)
                    else:
                        doc_lines.append(line)
                doc_lines.append('\n')

                out_lines.extend(lines[1:p])
                out_lines.append('\n')

        # 添加代码块
        code_content = [line for line in lines[p + 1:] if line.strip()]
        out_lines.append('```cpp\n')
        out_lines.extend(code_content)
        out_lines.append('\n```\n')

        if len(code_content) > 0:
            doc_lines.append('```cpp\n')
            doc_lines.extend(code_content)
            doc_lines.append('\n```\n')

        self.full_doc_lines.extend(doc_lines)
        if wf:
            self.wf_doc_lines.extend(doc_lines)

        return out_lines, code_content

    def write_markdown(self, path: Path, content: List[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(content), encoding='utf-8')

    def clean_extra_files(self):
        self.generated_files['index.md'] = True

        for file in self.website_dir.glob("*.md"):
            if file.name not in self.generated_files:
                print(f"[CLEAN] Removing extra file: {file.name}")
                file.unlink()

    def save_outputs(self):
        self.offline_dir.mkdir(parents=True, exist_ok=True)

        # 保存 print-all.md
        (self.offline_dir / 'print.md'   ).write_text(''.join(self.full_doc_lines), encoding='utf-8')
        (self.offline_dir / 'print-wf.md').write_text(''.join(self.  wf_doc_lines), encoding='utf-8')

        # 读取 mkdocs-header.yml
        header_config = {}
        if self.mkdocs_header.exists():
            with self.mkdocs_header.open('r', encoding='utf-8') as f:
                header_config = yaml.load(f, Loader=yaml.FullLoader) or {}

        # 合并 nav
        header_config['nav'] = self.nav_structure

        # 保存 mkdocs.yml
        with self.mkdocs_output.open('w', encoding='utf-8') as f:
            yaml.dump(header_config, f, sort_keys=False, allow_unicode=True)

    def process_directory(self, path: Path, layer: int = 1, parent_nav: List = None):
        """Travel directory tree"""
        if parent_nav is None:
            parent_nav = self.nav_structure

        for entry in sorted(path.iterdir()):
            if entry.is_dir():
                sub_nav = self.add_dir_to_nav(layer, entry.name, parent_nav)
                self.process_directory(entry, layer + 1, sub_nav)
            elif entry.is_file() and entry.suffix == '.cpp':
                self.add_file(layer, entry, parent_nav)

    def run(self):
        self.process_directory(self.template_dir)
        self.clean_extra_files()
        self.save_outputs()
    
def copy_static_files():
    if not STATIC_DIR.exists():
        print(f"[WARN] 静态目录不存在: {STATIC_DIR}")
        return
    for item in STATIC_DIR.iterdir():
        src = item
        dst = DOCS_DIR / item.name
        if src.is_dir():
            # copytree 不支持直接覆盖，需要先删除再复制
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    print(f"[INFO] 已将 {STATIC_DIR} 的内容复制到 {DOCS_DIR}")

def main():
    builder = MkDocsBuilder(
        CONFIG_PATH,
        TEMPLATE_DIR,
        WEBSITE_DIR,
        WEBSITE_WF_DIR,
        OFFLINE_DIR,
        MKDOCS_HEADER,
        MKDOCS_OUTPUT
    )
    builder.run()
    copy_static_files()

if __name__ == "__main__":
    main()
