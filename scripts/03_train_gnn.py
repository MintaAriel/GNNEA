"""Train cgcnn using the shared GNNEA training loop and configs/sirena.yml."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import config_from_cli, configure_runtime

if __name__ == '__main__':
    configure_runtime(config_from_cli())
    from src.training import training_main
    training_main('cgcnn')
