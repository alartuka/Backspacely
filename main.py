import asyncio
import json
import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl, validator, Field
import httpx
from github import Github, Repository, PullRequest, Issue
import git
import tempfile
from pathlib import Path

# initialize FastAPI app
app = FastAPI(title="Backspace AI Coding Agent", version="1.0.0")

# request model for the code endpoint
class CodeRequest(BaseModel):
    repoUrl: HttpUrl = Field(..., description="GitHub repository URL")
    prompt: str = Field(..., min_length=1, max_length=5000, description="Coding prompt")
    
    @validator('repoUrl')
    def validate_repo_url(cls, v):
        url_str = str(v)
        
        # Check URL length
        if len(url_str) > 500:
            raise ValueError("Repository URL is too long (max 500 characters)")
        
        # Ensure it's a GitHub URL
        if not url_str.startswith(('https://github.com/', 'git@github.com:')):
            raise ValueError("Only GitHub repositories are allowed")
        
        # Check for suspicious characters
        suspicious_chars = ['<', '>', '"', "'", '&', '|', ';', '`', '$', '(', ')']
        if any(char in url_str for char in suspicious_chars):
            raise ValueError("Repository URL contains suspicious characters")
        
        # Validate GitHub URL format
        github_pattern = r'^https://github\.com/[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+/?(?:\.git)?$'
        if not re.match(github_pattern, url_str.rstrip('/')):
            raise ValueError("Invalid GitHub repository URL format")
        
        return v
    
    @validator('prompt')
    def validate_prompt(cls, v):
        # Check for potentially harmful commands
        dangerous_patterns = [
            r'\brm\s+-rf\b',  # rm -rf command
            r'\bsudo\b',      # sudo command
            r'\bchmod\b',     # chmod command
            r'\bchown\b',     # chown command
            r'\b__import__\b', # Python __import__
            r'\beval\b',      # eval function
            r'\bexec\b',      # exec function
            r'\bos\.system\b', # os.system
            r'\bsubprocess\b', # subprocess module
            r'\bshell=True\b', # shell=True parameter
            r'[;&|`$(){}[\]]', # Shell metacharacters
            r'\.\./',         # Directory traversal
            r'/etc/',         # System directories
            r'/proc/',        # Process directories
            r'<script\b',     # Script tags
            r'javascript:',   # JavaScript protocol
            r'data:',         # Data protocol
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Prompt contains potentially harmful content")
        
        # Check for excessive special characters
        special_char_count = sum(1 for char in v if not char.isalnum() and not char.isspace())
        if special_char_count > len(v) * 0.3:  # More than 30% special characters
            raise ValueError("Prompt contains too many special characters")
        
        return v.strip()

@app.post("/code")
async def code_endpoint(request: CodeRequest):
    """Main endpoint to process coding requests with streaming updates"""
    
    # using token from env for GitHub authentication
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="GitHub token not configured")
    
    # initialize GitHub client
    github = Github(token)
    
    # validate GitHub token and permissions
    try:
        user = github.get_user()
        user.login  # raise exception if token is invalid
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid GitHub token")
    
    # check if token has necessary permissions for pull requests
    try:
        # test token permissions by checking user's repos (requires repo scope)
        list(user.get_repos(type="owner", per_page=1))
    except Exception as e:
        raise HTTPException(status_code=403, detail="GitHub token lacks necessary permissions for repository operations")
    
    # clone the repository
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