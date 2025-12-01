install_maturin:
    pip install maturin

build:
    uv run --python 3.10 maturin build --release
    uv run --python 3.11 maturin build --release
    uv run --python 3.12 maturin build --release
    uv run --python 3.13 maturin build --release
    uv run --python 3.14 maturin build --release
