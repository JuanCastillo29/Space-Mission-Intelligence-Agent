from pydantic_settings import BaseSettings


class GenerationSettings(BaseSettings):
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "llama-3.1-70b-versatile"
    LLM_FALLBACK_PROVIDER: str = "mistral"
    LLM_FALLBACK_MODEL: str = "mistral-large-latest"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    model_config = {"env_file": ".env", "extra": "ignore"}


gen_settings = GenerationSettings()
