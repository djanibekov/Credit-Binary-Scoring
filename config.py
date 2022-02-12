from starlette.config import Config
import os

config = Config('.env')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

APP_NAME = config('APP_NAME', cast=str)
# Debug mode
APP_DEBUG = config('APP_DEBUG', cast=bool, default=False)


ACCESS_TOKEN = config('ACCESS_TOKEN', cast=str)

LOGURU_LOG_FORMAT = config('LOGURU_LOG_FORMAT')
LOGGING_LOG_FORMAT = config('LOGGING_LOG_FORMAT')

SENTRY_DSN = config('SENTRY_DSN')
