from pathlib import Path

path = Path(__file__).absolute().parent.parent
print("Working directory:", path)
model_dir = path / 'models'
if not model_dir.exists():
    model_dir.mkdir()

data_dir = path.parent.parent / 'data'
print("Dataset directory:", data_dir)
