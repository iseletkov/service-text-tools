import uvicorn
from fastapi import FastAPI
from app.routers import CControllerTextUtils

app = FastAPI()
app.include_router(CControllerTextUtils.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

