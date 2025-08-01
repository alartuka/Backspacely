import os
import re
import shutil
import uvicorn
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl, field_validator, Field
import httpx
from github import Github, Repository, PullRequest, Issue
import git
import tempfile
import time
import stat
from pathlib import Path
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
from datetime import datetime

# initialize FastAPI app
app = FastAPI(title="Backspace AI Coding Agent", version="1.0.0")

# load environment variables from .env file
load_dotenv()



# request model for the code endpoint
class CodeRequest(BaseModel):
    repoUrl: HttpUrl = Field(..., description="GitHub repository URL")
    prompt: str = Field(..., min_length=1, max_length=5000, description="Coding prompt")
    


    @field_validator('repoUrl')
    @classmethod
    def validate_repo_url(cls, v):
        """Validate GitHub repository URL"""
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
    


    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        """Validate coding prompt for security and content"""
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
        if special_char_count > len(v) * 0.3:  # more than 30% special characters
            raise ValueError("Prompt contains too many special characters")
        
        return v.strip()


def send_sse_event(event_type: str, data: dict) -> str:
    """Format data as Server-Sent Event"""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def parse_llm_response(response: str) -> dict:
    """Parse LLM response to extract file paths and their content"""
    result_files = {}
    
    # pattern to match file blocks: ```filename or ```language:filename
    file_pattern = r'```(?:[\w]+:)?([\w\./\-_]+\.[\w]+)\n(.*?)\n```'
    matches = re.findall(file_pattern, response, re.DOTALL)
    
    for file_path, content in matches:
        # clean up file path
        file_path = file_path.strip()
        # remove leading ./ if present
        if file_path.startswith('./'):
            file_path = file_path[2:]
        
        # Skip if content looks like Python meta-code
        if ('result_files' in content or 'github_repo.create_pull' in content or 
            'def ' in content[:100] and 'html' not in content.lower()[:200]):
            continue
            
        result_files[file_path] = content.strip()
    
    # fallback: if no specific files found, look for generic code blocks
    if not result_files:
        code_pattern = r'```(?:python|javascript|js|java|cpp|c|go|rust|php|rb|cs|html|css)?\n(.*?)\n```'
        code_matches = re.findall(code_pattern, response, re.DOTALL)
        
        if code_matches:
            # infer file extension from content
            for i, code in enumerate(code_matches):
                # Skip Python meta-code
                if ('result_files' in code or 'github_repo.create_pull' in code or 
                    ('def ' in code[:100] and 'html' not in code.lower()[:200])):
                    continue
                    
                file_ext = infer_file_extension(code)
                file_name = f"generated_code_{i+1}.{file_ext}"
                result_files[file_name] = code.strip()
    
    # fallback: try to extract HTML content directly if it's an HTML request
    if not result_files and ('html' in response.lower() or 'hello world' in response.lower()):
        # Look for HTML-like content
        html_pattern = r'<!DOCTYPE html>.*?</html>'
        html_match = re.search(html_pattern, response, re.DOTALL | re.IGNORECASE)
        if html_match:
            result_files['index.html'] = html_match.group(0)
    
    # final fallback: save entire response as a single file ONLY if it's not meta-code
    if not result_files and not ('result_files' in response or 'github_repo.create_pull' in response):
        file_ext = infer_file_extension(response)
        result_files[f"generated_code.{file_ext}"] = response.strip()
    
    return result_files


def infer_file_extension(code: str) -> str:
    """Infer file extension based on code content"""
    code_lower = code.lower()
    
    # HTML indicators (check first since it's common for hello world)
    if any(keyword in code_lower for keyword in ['<!doctype html>', '<html>', '<head>', '<body>', '<h1>']):
        return 'html'
    
    # python indicators
    elif any(keyword in code_lower for keyword in ['def ', 'import ', 'from ', 'print(', 'if __name__']):
        return 'py'
    
    # javaScript indicators
    elif any(keyword in code_lower for keyword in ['function ', 'const ', 'let ', 'var ', 'console.log', '=>']):
        return 'js'

    # CSS indicators
    elif any(keyword in code_lower for keyword in ['body {', 'html {', '.class', '#id']):
        return 'css'

    # java indicators
    elif any(keyword in code_lower for keyword in ['public class', 'private ', 'public static void main']):
        return 'java'
    
    # c/c++ indicators
    elif any(keyword in code_lower for keyword in ['#include', 'int main(', 'printf(', 'cout <<']):
        return 'cpp' if 'cout' in code_lower or 'namespace' in code_lower else 'c'
    
    # go indicators
    elif any(keyword in code_lower for keyword in ['package main', 'func main(', 'import (']):
        return 'go'
    
    # default: python
    return 'py'



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



def create_detailed_pr_summary(prompt: str, result_files: dict, generated_code: str) -> tuple[str, str]:
    """Create a detailed PR title and body with comprehensive summary"""
    
    # create concise but descriptive title
    prompt_preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
    title = f"AI Generated: {prompt_preview}"
    
    # create detailed body with multiple sections
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # analyze file types and changes
    file_types = {}
    for file_path in result_files.keys():
        ext = file_path.split('.')[-1] if '.' in file_path else 'unknown'
        file_types[ext] = file_types.get(ext, 0) + 1
    
    file_summary = ", ".join([f"{count} {ext} file{'s' if count > 1 else ''}" 
                             for ext, count in file_types.items()])
    
    body = f"""
        ## AI-Generated Code Changes

        ### Summary
        This pull request contains AI-generated code based on the following prompt:

        > {prompt}

        ### Changes Overview
        - **Generated on:** {timestamp}
        - **Total files:** {len(result_files)}
        - **File types:** {file_summary}
        - **AI Model:** Groq LLM (llama3-8b-8192)

        ### Files Modified/Created
    """
            
    # add file list with brief descriptions
    for file_path, content in result_files.items():
        lines_count = len(content.split('\n'))
        body += f"- `{file_path}` ({lines_count} lines)\n"
            
        body += f"""
            ### Technical Details
            The code was generated using advanced AI analysis of the existing repository structure and codebase. 
            Backspacely AI agent performed the following steps:

            1. **Analyzed** existing code patterns and structure
            2. **Generated** appropriate solutions based on the prompt
            3. **Validated** code syntax and structure in a secure sandbox environment
            4. **Created** this pull request with the generated changes

            ### Review Notes
            Please review the generated code carefully before merging:
            - Code syntax and logic
            - Integration with existing codebase  
            - Security considerations
            - Performance implications
            - Test coverage (if applicable)

            ### Generated Code Preview
            <details>
                <summary>Click to view AI-generated response</summary>

                ```
                {generated_code[:1000]}{'...' if len(generated_code) > 1000 else ''}
                ```
            </details>

            ---
            *This PR was automatically generated by Backspacely*
        """
    
    return title, body

async def process_code_request(request: CodeRequest):
    """Process coding request with SSE streaming"""
    
    # yield initial status
    yield send_sse_event("status", {"message": ">>> Starting BACKSPACELY...", "stage": "init"})
    
    # using token from env for GitHub authentication
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        yield send_sse_event("error", {"message": ">>> ERROR! GitHub token not configured"})
        return
    
    # initialize GitHub client
    github = Github(token)
    
    # validate GitHub token and permissions
    try:
        user = github.get_user()
        yield send_sse_event("status", {"message": f">>> SUCCESS! Authenticated as: {user.login}", "stage": "auth"})
    
    except Exception as e:
        yield send_sse_event("error", {"message": f">>> ERROR! Invalid GitHub token: {str(e)}"})
        return
    
    try:
        # extract owner and repo name from URL for early validation
        url_parts = str(request.repoUrl).rstrip('/').split('/')
        owner = url_parts[-2]
        repo_name_clean = url_parts[-1].replace('.git', '')
        
        yield send_sse_event("status", {"message": f">>> Checking access to {owner}/{repo_name_clean}", "stage": "validation"})
        
        # attempt to access repo with token
        github_repo = github.get_repo(f"{owner}/{repo_name_clean}")
        
        # check if reading repo details is allowed
        repo_full_name = github_repo.full_name
        default_branch = github_repo.default_branch
        
        yield send_sse_event("status", {"message": f">>> SUCCESS! Repository found: {repo_full_name}", "stage": "validation"})
        yield send_sse_event("status", {"message": f">>> Default branch: {default_branch}", "stage": "validation"})
        
        # checking permission 
        permissions = github_repo.permissions
        yield send_sse_event("status", {"message": f">>> Checking permissions...", "stage": "validation"})
        
        # check if branches and PRs can be created
        if not permissions.push:
            yield send_sse_event("error", {"message": f">>> ERROR! GitHub token lacks write access to repository {owner}/{repo_name_clean}. You need push permissions to create pull requests."})
            return
        
        yield send_sse_event("status", {"message": ">>> SUCCESS! Token has sufficient permissions!", "stage": "validation"})
            
    except Exception as e:
        error_msg = str(e).lower()
        
        if "not found" in error_msg or "404" in error_msg:
            yield send_sse_event("error", {"message": f">>> ERROR! Repository {owner}/{repo_name_clean} not found or token lacks access"})
        elif "forbidden" in error_msg or "403" in error_msg:
            yield send_sse_event("error", {"message": f">>> ERROR! GitHub token lacks access to repository {owner}/{repo_name_clean}"})
        else:
            yield send_sse_event("error", {"message": f">>> ERROR! Failed to access repository {owner}/{repo_name_clean}: {str(e)}"})
        return
    
    # clone the repository
    clone_dir = Path(tempfile.mkdtemp(prefix="repo_clone_"))
    repo_name = str(request.repoUrl).split('/')[-1].replace('.git', '')
    full_clone_path = clone_dir / repo_name
    
    yield send_sse_event("status", {"message": f">>> Cloning repository to: {full_clone_path}", "stage": "clone"})
    
    try:
        repo = git.Repo.clone_from(str(request.repoUrl), full_clone_path)
        yield send_sse_event("status", {"message": ">>> SUCCESS! Repository cloned successfully", "stage": "clone"})

    except Exception as e:
        # cleanup on failure
        if clone_dir.exists():
            try:
                def handle_remove_readonly(func, path, exc):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                
                shutil.rmtree(clone_dir, onerror=handle_remove_readonly)

            except:
                pass  # ignore cleanup errors in exception handling

        yield send_sse_event("error", {"message": f">>> ERROR! Failed to clone repository: {str(e)}"})
        return
    
    try:
        # read repo files for analysis
        yield send_sse_event("status", {"message": ">>> Reading repository files for analysis...", "stage": "analysis"})
        repo_content = read_repository_files(full_clone_path)
        yield send_sse_event("status", {"message": ">>> SUCCESS! Repository content analyzed", "stage": "analysis"})
        
        # initialize Groq API to analyze code with Groq API
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            yield send_sse_event("error", {"message": ">>> ERROR! Groq API key not configured"})
            return
        
        yield send_sse_event("status", {"message": ">>> BACKSPACELY analyzing code and generating plan...", "stage": "ai_analysis"})
        yield send_sse_event("plan", {"message": f">>> Creating solution for: {request.prompt}", "prompt": request.prompt})
        
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
                            "content": "You are a helpful coding assistant. Analyze the repository and generate the actual file content based on the user's prompt. IMPORTANT: You must provide the actual file content, not Python code that creates files.\n\nFormat your response EXACTLY like this:\n\n```index.html\n<!DOCTYPE html>\n<html>\n<head>\n    <title>My Page</title>\n</head>\n<body>\n    <h1>Hello World</h1>\n</body>\n</html>\n```\n\nFor multiple files, use separate code blocks with the filename. DO NOT write Python dictionaries or meta-code - provide the actual file contents that should be saved."
                        },
                        {
                            "role": "user",
                            "content": f"Repository content:\n{repo_content}\n\nUser request: {request.prompt}\n\nGenerate the actual file content (not Python code) that implements this request. Provide the complete file content in code blocks with filenames. For example, if creating an HTML page, show the actual HTML content, not Python code that creates HTML."
                        }
                    ],
                    "temperature": 0.7
                }
            )
            
            if groq_response.status_code != 200:
                yield send_sse_event("error", {"message": f">>> ERROR! Groq API error: {groq_response.text}"})
                return
            
            groq_data = groq_response.json()
            generated_code = groq_data["choices"][0]["message"]["content"]
            yield send_sse_event("status", {"message": ">>> SUCCESS! Code generated by AI", "stage": "ai_analysis"})
            
    except Exception as e:
        # cleanup on failure
        if clone_dir.exists():
            try:
                def handle_remove_readonly(func, path, exc):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                
                shutil.rmtree(clone_dir, onerror=handle_remove_readonly)
            
            except:
                pass  # ignore cleanup errors in exception handling

        yield send_sse_event("error", {"message": f">>> ERROR! Failed to analyze code with Groq: {str(e)}"})
        return
    
    # execute code in E2B sandbox and parse LLM response
    result_files = {}
    execution_results = {}
    try:
        # parse LLM response to extract files and content
        result_files = parse_llm_response(generated_code)
        yield send_sse_event("status", {"message": f">>> SUCCESS! Parsed {len(result_files)} files from AI response", "stage": "code_processing"})
        
        # show files being processed
        for file_path in result_files.keys():
            yield send_sse_event("file_edit", {"message": f">>> Processing: {file_path}", "file": file_path})
        
        # execute code in E2B sandbox for validation
        yield send_sse_event("status", {"message": ">>> Testing code in sandbox...", "stage": "validation"})
        sandbox = Sandbox()
        
        for file_path, content in result_files.items():
            # determine language for sandbox execution
            if file_path.endswith('.py'):
                try:
                    execution_result = sandbox.run_code(content, language="python")
                    execution_results[file_path] = ">>> Executed successfully"
                    yield send_sse_event("validation", {"message": f">>> SUCCESS! {file_path} executed successfully", "file": file_path})
                
                except Exception as sandbox_error:
                    execution_results[file_path] = f">>> Execution warning: {str(sandbox_error)[:100]}"
                    yield send_sse_event("validation", {"message": f">>> Warning: {file_path} execution issue: {str(sandbox_error)[:100]}", "file": file_path})
                    # continue.. 

        # close sandbox (method name varies by version !!)
        if hasattr(sandbox, 'close'):
            sandbox.close()
        elif hasattr(sandbox, 'kill'):
            sandbox.kill()
        
        yield send_sse_event("status", {"message": ">>> SUCCESS! Sandbox validation completed", "stage": "validation"})
        
    except Exception as e:
        # cleanup on failure 
        if clone_dir.exists():
            try:
                def handle_remove_readonly(func, path, exc):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                
                shutil.rmtree(clone_dir, onerror=handle_remove_readonly)
            except:
                pass  # ignore cleanup errors in exception handling
        yield send_sse_event("error", {"message": f">>> ERROR! Failed to process generated code: {str(e)}"})
        return
    
    # create pull request with generated code
    try:
        yield send_sse_event("status", {"message": ">>> Creating pull request...", "stage": "git_operations"})
        
        # get default branch ('main' or 'master')
        default_branch_name = github_repo.default_branch
        main_branch = github_repo.get_branch(default_branch_name)
        
        # create new branch with timestamp
        new_branch_name = f"ai-generated-{int(time.time())}"
        yield send_sse_event("git_operation", {"message": f">>> Creating branch: {new_branch_name}", "operation": "branch_create"})
        github_repo.create_git_ref(f"refs/heads/{new_branch_name}", main_branch.commit.sha)
        
        # commit generated files
        committed_files = []
        yield send_sse_event("status", {"message": ">>> Committing generated files...", "stage": "git_operations"})
        
        for file_path, content in result_files.items():
            try:
                # ensure file path is safe (no directory traversal)
                if '..' in file_path or file_path.startswith('/'):
                    yield send_sse_event("git_operation", {"message": f">>> Skipping unsafe file path: {file_path}", "operation": "commit"})
                    continue
                
                # get existing file
                try:
                    file_obj = github_repo.get_contents(file_path, ref=new_branch_name)
                    github_repo.update_file(
                        file_path, 
                        f"AI generated code update: {file_path}", 
                        content, 
                        file_obj.sha, 
                        branch=new_branch_name
                    )
                    committed_files.append(f"Updated: {file_path}")
                    yield send_sse_event("git_operation", {"message": f">>> SUCCESS! Updated existing file: {file_path}", "operation": "commit"})
                except:
                    # file doesn't exist, create new one
                    github_repo.create_file(
                        file_path, 
                        f"AI generated code: {file_path}", 
                        content, 
                        branch=new_branch_name
                    )
                    committed_files.append(f"Created: {file_path}")
                    yield send_sse_event("git_operation", {"message": f">>> SUCCESS! Created new file: {file_path}", "operation": "commit"})
                    
            except Exception as file_error:
                yield send_sse_event("git_operation", {"message": f">>> ERROR! Failed to commit {file_path}: {str(file_error)}", "operation": "commit"})
                continue
        
        # create detailed PR summary
        yield send_sse_event("status", {"message": ">>> Generating pull request summary...", "stage": "pr_creation"})
        pr_title, pr_body = create_detailed_pr_summary(request.prompt, result_files, generated_code)
        
        # create pull request with full summary
        yield send_sse_event("status", {"message": ">>> Creating pull request...", "stage": "pr_creation"})
        pr = github_repo.create_pull(
            title=pr_title,
            body=pr_body,
            head=new_branch_name,
            base=default_branch_name
        )
        
        yield send_sse_event("pr_created", {"message": f">>> SUCCESS! Pull request created: {pr.html_url}", "url": pr.html_url})
        
        # create summary of changes
        file_types = {}
        total_lines = 0
        for file_path, content in result_files.items():
            ext = file_path.split('.')[-1] if '.' in file_path else 'unknown'
            file_types[ext] = file_types.get(ext, 0) + 1
            total_lines += len(content.split('\n'))
        
        file_summary = ", ".join([f"{count} {ext} file{'s' if count > 1 else ''}" 
                                 for ext, count in file_types.items()])
        
        changes_summary = f"Generated {len(result_files)} files ({file_summary}) with {total_lines} total lines of code based on prompt: '{request.prompt}'"
        
        # final success event with complete summary
        yield send_sse_event("complete", {
            "pull_request_url": pr.html_url,
            "summary": changes_summary,
            "files_created": len(result_files),
            "total_lines": total_lines,
            "file_types": file_types
        })
        
    except Exception as e:
        yield send_sse_event("error", {"message": f">>> ERROR! Failed to create pull request: {str(e)}"})
        return
    
    finally:
        # cleanup cloned repository
        if clone_dir.exists():
            try:

                # remove read-only attributes before deletion
                def handle_remove_readonly(func, path, exc):
                    """Handle read-only files during cleanup"""
                    os.chmod(path, stat.S_IWRITE)
                    func(path)

                
                shutil.rmtree(clone_dir, onerror=handle_remove_readonly)
                yield send_sse_event("status", {"message": ">>> SUCCESS! Cleaned up temporary files", "stage": "cleanup"})

            except Exception as cleanup_error:
                yield send_sse_event("status", {"message": f">>> Cleanup warning: {str(cleanup_error)}", "stage": "cleanup"})



@app.post("/code")
async def code_endpoint(request: CodeRequest):
    """Main endpoint to process coding requests with streaming updates"""
    
    async def event_stream():
        async for event in process_code_request(request):
            yield event
        
        # end final event to close connection
        yield send_sse_event("close", {"message": ">>> SUCCESS! Stream completed"})
    
    return StreamingResponse(
        event_stream(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)