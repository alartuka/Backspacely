import asyncio
import json
import os
import re
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl, validator, Field
import httpx
from github import Github, Repository, PullRequest, Issue
import git
import tempfile
from pathlib import Path
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv

# initialize FastAPI app
app = FastAPI(title="Backspace AI Coding Agent", version="1.0.0")

# load environment variables from .env file
load_dotenv()

# request model for the code endpoint
class CodeRequest(BaseModel):
    repoUrl: HttpUrl = Field(..., description="GitHub repository URL")
    prompt: str = Field(..., min_length=1, max_length=5000, description="Coding prompt")
    
    @validator('repoUrl')
    def validate_repo_url(cls, v):
        url_str = str(v)
        
        # check URL length
        if len(url_str) > 500:
            raise ValueError("Repository URL is too long (max 500 characters)")
        
        # ensure it's a GitHub URL
        if not url_str.startswith(('https://github.com/', 'git@github.com:')):
            raise ValueError("Only GitHub repositories are allowed")
        
        # check for suspicious characters
        suspicious_chars = ['<', '>', '"', "'", '&', '|', ';', '`', '$', '(', ')']
        if any(char in url_str for char in suspicious_chars):
            raise ValueError("Repository URL contains suspicious characters")
        
        # validate GitHub URL format
        github_pattern = r'^https://github\.com/[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+/?(?:\.git)?$'
        if not re.match(github_pattern, url_str.rstrip('/')):
            raise ValueError("Invalid GitHub repository URL format")
        
        return v
    
    @validator('prompt')
    def validate_prompt(cls, v):
        # check for potentially harmful commands
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
        
        # check for excessive special characters
        special_char_count = sum(1 for char in v if not char.isalnum() and not char.isspace())
        if special_char_count > len(v) * 0.3:  # More than 30% special characters
            raise ValueError("Prompt contains too many special characters")
        
        return v.strip()

def read_repository_files(repo_path: Path, max_files: int = 20) -> str:
    """Read and summarize repository files for LLM analysis"""
    file_contents = []
    file_count = 0
    
    # common code file extensions
    code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs'}
    
    for file_path in repo_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in code_extensions:
            if file_count >= max_files:
                break
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    relative_path = file_path.relative_to(repo_path)
                    file_contents.append(f"File: {relative_path}\n{content}\n{'='*50}\n")
                    file_count += 1
            except Exception as e:
                continue
    
    return '\n'.join(file_contents) if file_contents else "No readable code files found."

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
    clone_dir = Path(tempfile.mkdtemp(prefix="repo_clone_"))
    repo_name = str(request.repoUrl).split('/')[-1].replace('.git', '')
    full_clone_path = clone_dir / repo_name
    
    try:
        repo = git.Repo.clone_from(str(request.repoUrl), full_clone_path)
    except Exception as e:
        # cleanup on failure
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        raise HTTPException(status_code=400, detail=f"Failed to clone repository: {str(e)}")
    
    try:
        # read repository files for analysis
        repo_content = read_repository_files(full_clone_path)
        
        # analyze code with Groq API 
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="Groq API key not configured")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            groq_response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful coding assistant. Analyze the repository and generate code based on the user's prompt. Return only the code that should be created or modified, with clear file paths."
                        },
                        {
                            "role": "user",
                            "content": f"Repository content:\n{repo_content}\n\nUser prompt: {request.prompt}\n\nGenerate the necessary code changes."
                        }
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.7
                }
            )
            
            if groq_response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Groq API error: {groq_response.text}")
            
            groq_data = groq_response.json()
            generated_code = groq_data["choices"][0]["message"]["content"]
            
    except Exception as e:
        # cleanup on failure
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        raise HTTPException(status_code=500, detail=f"Failed to analyze code with Groq: {str(e)}")
    
    # execute code in E2B sandbox
    result_files = {}
    try:
        sandbox = Sandbox()
        
        # create a simple Python script to execute the generated code
        execution_script = f"""# Generated code execution {generated_code}"""
        
        # execute in sandbox
        execution_result = sandbox.run_code(execution_script, language="python")
        
        # for now: saving generated code as a new file
        # ! later: need to parse LLM response to determine file paths
        result_files["generated_code.py"] = generated_code
        
        sandbox.close()
        
    except Exception as e:
        # cleanup on failure
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        raise HTTPException(status_code=500, detail=f"Failed to execute code in sandbox: {str(e)}")
    
    # create pull request with generated code
    try:
        # extract owner and repo name from URL
        url_parts = str(request.repoUrl).rstrip('/').split('/')
        owner = url_parts[-2]
        repo_name_clean = url_parts[-1].replace('.git', '')
        
        github_repo = github.get_repo(f"{owner}/{repo_name_clean}")
        
        # get default branch ('main' or 'master')
        default_branch_name = github_repo.default_branch
        main_branch = github_repo.get_branch(default_branch_name)
        
        # create new branch with timestamp
        import time
        new_branch_name = f"ai-generated-{int(time.time())}"
        github_repo.create_git_ref(f"refs/heads/{new_branch_name}", main_branch.commit.sha)
        
        # commit generated files
        for file_path, content in result_files.items():
            try:
                # try to get existing file
                file_obj = github_repo.get_contents(file_path, ref=new_branch_name)
                github_repo.update_file(
                    file_path, 
                    "AI generated code update", 
                    content, 
                    file_obj.sha, 
                    branch=new_branch_name
                )
            except:
                # file doesn't exist, create new one
                github_repo.create_file(
                    file_path, 
                    "AI generated code", 
                    content, 
                    branch=new_branch_name
                )
        
        # create pull request
        pr = github_repo.create_pull(
            title=f"AI Generated Code: {request.prompt[:50]}...",
            body=f"Generated code based on prompt: {request.prompt}\n\nGenerated using Groq LLM (llama3-8b-8192)",
            head=new_branch_name,
            base=default_branch_name
        )
        
        return {
            "status": "success", 
            "pull_request_url": pr.html_url,
            "branch_name": new_branch_name,
            "files_created": list(result_files.keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create pull request: {str(e)}")
    
    finally:
        # cleanup cloned repository
        if clone_dir.exists():
            shutil.rmtree(clone_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)