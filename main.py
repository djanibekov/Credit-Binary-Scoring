from loguru import logger
from fastapi import FastAPI, Depends, Request

import os
import logging
import sentry_sdk
import log.handlers

import config
from routers import middleware, scoring


def pre_start_configuration():
    sentry_sdk.init(
        dsn=config.SENTRY_DSN,
        traces_sample_rate=1.0
    )

    if not os.path.exists('storage/logs'):
        os.makedirs('storage/logs')

    logger.add('storage/logs/logs-{time:YYYY-MM-DD}.log', level='INFO', format=config.LOGURU_LOG_FORMAT,
               rotation='00:00')
    logging.basicConfig(handlers=[log.handlers.InterceptHandler()], level=logging.INFO)


app = FastAPI(debug=config.APP_DEBUG)

app.include_router(
    scoring.router,
    prefix='/scoring',
    # dependencies=[Depends(middleware.access_token_header)]
)


@app.middleware('http')
async def sentry_handler(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        with sentry_sdk.push_scope() as scope:
            scope.set_context('request', request)
            scope.user = {
                'ip_address': request.client.host
            }
            sentry_sdk.capture_exception(e)
        raise e

pre_start_configuration()
