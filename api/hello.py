from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()


@app.get("/")
def hello():
    """Minimal serverless function to smoke-test Vercel runtime."""
    return JSONResponse({"message": "Hello from Vercel!", "status": "ok"})


# Vercel serverless function handler
def handler(request, response):
    return app
