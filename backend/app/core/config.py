import os


class Settings:
    """
    Application configuration.

    Environment variables override local defaults.
    """

    app_name: str = os.getenv(
        "THREATGUARD_APP_NAME",
        "ThreatGuard",
    )

    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://"
        "threatguard:threatguard"
        "@localhost:5432/threatguard",
    )


settings = Settings()
