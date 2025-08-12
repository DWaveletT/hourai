#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

AUX_DIR = ROOT_DIR / 'aux'
PRINT_DIR = ROOT_DIR / 'docs' / 'offline'
HOURAI_TEX = ROOT_DIR / 'utils' / 'hourai.tex'

FILES = ['print', 'print-wf']

MD_FILE1 = PRINT_DIR / 'print.md'
MD_FILE2 = PRINT_DIR / 'print-wf.md'
PDF_FILE1 = PRINT_DIR / 'print.pdf'
PDF_FILE2 = PRINT_DIR / 'print-wf.pdf'

def build_pdf(md_file: Path, tex_file: Path, pdf_file: Path):
    if not md_file.exists():
        print(f"[ERROR] Markdown file not exists: {md_file}")
        sys.exit(1)

    if not HOURAI_TEX.exists():
        print(f"[ERROR] LaTeX template not exists: {HOURAI_TEX}")
        sys.exit(1)

    # 确保输出目录存在
    PRINT_DIR.mkdir(parents=True, exist_ok=True)

    cmd_tex = [
        "pandoc", str(md_file),
        f"--output={str(tex_file)}",
        f"--template={str(HOURAI_TEX)}",
        "--pdf-engine=xelatex",
        "--listing"
    ]

    # 2. 调用 latexmk 生成 PDF
    cmd_pdf = [
        "latexmk",
        "-pdf",
        "-xelatex",
        f"-output-directory={str(PRINT_DIR)}",
        "-interaction=nonstopmode",
        str(tex_file)
    ]

    # 3. 调用 latexmk 清理文件
    cmd_clear = [
        "latexmk",
        "-pdf",
        "-xelatex",
        f"-output-directory={str(PRINT_DIR)}",
        "-c",
        "-interaction=nonstopmode",
        str(tex_file)
    ]

    try:
        print("[INFO] Running Pandoc to generate TEX...")
        print(" ".join(cmd_tex))
        subprocess.run(cmd_tex, check=True)

        print("[INFO] Running LaTeXMK to generate PDF...")
        print(" ".join(cmd_pdf))
        subprocess.run(cmd_pdf, check=True)

        print("[INFO] Running LaTeXMK to clear up...")
        print(" ".join(cmd_clear))
        subprocess.run(cmd_clear, check=True)

        print(f"[SUCCESS] Generate finished: {pdf_file}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Generate failed, exit code: {e.returncode}")
        sys.exit(e.returncode)

def main():
    for file in FILES:
        md_file  = PRINT_DIR / (file + ".md")
        tex_file = PRINT_DIR / (file + ".tex")
        pdf_file = PRINT_DIR / (file + ".pdf")
        build_pdf(md_file, tex_file, pdf_file)

if __name__ == "__main__":
    main()
