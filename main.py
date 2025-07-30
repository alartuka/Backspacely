from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from fastapi.responses import StreamingResponse
import git
from github import Github
import json
import os

app = FastAPI(title="Backspacly", version="1.0.0")


# initialize GitHub client
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
github_client = Github(GITHUB_TOKEN)


@app.post("/code")
async def bsly_agent(request: ):
    """Main endpoint for Backspacly"""