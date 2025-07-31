import asyncio
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
import httpx
from github import Github, Repository, PullRequest, Issue
import git
import tempfile
from pathlib import Path

# initialize FastAPI app
app = FastAPI(title="Backspace AI Coding Agent", version="1.0.0")

# request model for the code endpoint
class CodeRequest(BaseModel):
    repoUrl: HttpUrl
    prompt: str


@app.post("/code")
async def code_endpoint(request: CodeRequest):
    """Main endpoint to process coding requests with streaming updates"""
    
    # using token from env for GitHub authentication
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="GitHub token not configured")
    
    # initialize GitHub client
    github = Github(token)

    # Clone the repository
    clone_dir = Path(tempfile.mkdtemp(prefix="repo_clone_")) # create a temporary directory for cloning
    repo_name = str(request.repoUrl).split('/')[-1].replace('.git', '') # extract repo name from URL
    full_clone_path = clone_dir / repo_name # full path for cloned repo

    try:
        repo = git.Repo.clone_from(str(request.repoUrl), full_clone_path) # clone repo
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to clone repository: {str(e)}")
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)