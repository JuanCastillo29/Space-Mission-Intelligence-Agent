from pathlib import Path

from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class EvalSettings(BaseSettings):
    EVAL_DATASET_PATH: str = str(
        _PROJECT_ROOT / "scripts" / "evaluation" / "data" / "golden_qa.json"
    )
    EVAL_RESULTS_DIR: str = str(
        _PROJECT_ROOT / "scripts" / "evaluation" / "results"
    )
    EVAL_RETRIEVAL_K_VALUES: list[int] = [3, 5, 10]
    EVAL_RERANK_TOP_K: int = 10
    EVAL_FINAL_TOP_K: int = 5
    EVAL_RAGAS_ENABLED: bool = True
    EVAL_BATCH_SIZE: int = 5

    model_config = {"env_file": ".env", "extra": "ignore"}


eval_settings = EvalSettings()
