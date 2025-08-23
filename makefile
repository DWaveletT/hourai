all: docs print
	python scripts/build_docs.py
	python scripts/build_print.py

docs: scripts/build_docs.py mkdocs.header.yml
	python scripts/build_docs.py

print: scripts/build_print.py hourai.tex
	python scripts/build_print.py