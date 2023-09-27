import appserver.routes

from fastapi import FastAPI
app = FastAPI()

app.include_router(routes.router)
