from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_CORS_ORIGINS: list[str] = ["http://localhost:8501"]
    API_TITLE: str = "Space Mission Intelligence Agent"
    API_VERSION: str = "v1"
    LOG_LEVEL: str = "INFO"

    model_config = {"env_file": ".env", "extra": "ignore"}


api_settings = APISettings()
