from fastapi import Header, HTTPException
from typing import Optional

import config


async def access_token_header(x_access_token_header: Optional[str] = Header(None)):
    if x_access_token_header != config.ACCESS_TOKEN:
        raise HTTPException(status_code=403, detail='Invalid access token')
