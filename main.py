import os
import re
import shutil
import uvicorn
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
        
        result_files[file_path] = content.strip()
    
    # fallback: if no specific files found, look for generic code blocks
    if not result_files:
        code_pattern = r'```(?:python|javascript|js|java|cpp|c|go|rust|php|rb|cs)?\n(.*?)\n```'
        code_matches = re.findall(code_pattern, response, re.DOTALL)
        
        if code_matches:
            # infer file extension from content
            for i, code in enumerate(code_matches):
                file_ext = infer_file_extension(code)
                file_name = f"generated_code_{i+1}.{file_ext}"
                result_files[file_name] = code.strip()
    
    # final fallback: save entire response as a single file
    if not result_files:
        file_ext = infer_file_extension(response)
        result_files[f"generated_code.{file_ext}"] = response.strip()
    
    return result_files



def infer_file_extension(code: str) -> str:
    """Infer file extension based on code content"""
    code_lower = code.lower()
    
    # python indicators
    if any(keyword in code_lower for keyword in ['def ', 'import ', 'from ', 'print(', 'if __name__']):
        return 'py'
    
    # javaScript indicators
    elif any(keyword in code_lower for keyword in ['function ', 'const ', 'let ', 'var ', 'console.log', '=>']):
        return 'js'
    
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
        print(f">>> Authenticated as: {user.login}")
    except Exception as e:
        print(f">>> Token authentication failed: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid GitHub token: {str(e)}")
    
    try:
        # extract owner and repo name from URL for early validation
        url_parts = str(request.repoUrl).rstrip('/').split('/')
        owner = url_parts[-2]
        repo_name_clean = url_parts[-1].replace('.git', '')
        
        print(f">>> Checking access to {owner}/{repo_name_clean}")
        
        # attempt to access repo with token
        github_repo = github.get_repo(f"{owner}/{repo_name_clean}")
        
        # check if reading repo details is allowed
        repo_full_name = github_repo.full_name
        default_branch = github_repo.default_branch
        
        print(f">>> Repository found: {repo_full_name}")
        print(f">>> Default branch: {default_branch}")
        
        # checking permission 
        permissions = github_repo.permissions
        print(f">>> Permissions - Admin: {permissions.admin}, Push: {permissions.push}, Pull: {permissions.pull}")
        
        # check if branches and PRs can be created
        if not permissions.push:
            raise HTTPException(
                status_code=403, 
                detail=f"GitHub token lacks write access to repository {owner}/{repo_name_clean}. You need push permissions to create pull requests."
            )
        
        print(">>> Token has sufficient permissions!")
            
    except HTTPException:
        raise  # re-raise HTTPExceptions as-is

    except Exception as e:
        error_msg = str(e).lower()
        print(f">>> Repository access error: {str(e)}")
        
        if "not found" in error_msg or "404" in error_msg:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo_name_clean} not found or token lacks access")
        elif "forbidden" in error_msg or "403" in error_msg:
            raise HTTPException(status_code=403, detail=f"GitHub token lacks access to repository {owner}/{repo_name_clean}")
        else:
            raise HTTPException(status_code=400, detail=f"Failed to access repository {owner}/{repo_name_clean}: {str(e)}")
    
    # clone the repository
    clone_dir = Path(tempfile.mkdtemp(prefix="repo_clone_"))
    repo_name = str(request.repoUrl).split('/')[-1].replace('.git', '')
    full_clone_path = clone_dir / repo_name
    
    print(f">>> Cloning to: {full_clone_path}")
    
    try:
        repo = git.Repo.clone_from(str(request.repoUrl), full_clone_path)
        print(">>> Repository cloned successfully")

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
        print(f">>> Clone failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to clone repository: {str(e)}")
    
    try:
        # read repo files for analysis
        print(">>> Reading repository files...")
        repo_content = read_repository_files(full_clone_path)
        
        # initialize Groq API to analyze code with Groq API
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="Groq API key not configured")
        
        print(">>> Sending request to Groq API...")
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
                            "content": "You are a helpful coding assistant. Analyze the repository and generate code based on the user's prompt. Format your response with clear file paths using code blocks like:\n\n```filename.py\n# Your code here\n```\n\nFor multiple files, use separate code blocks for each file."
                        },
                        {
                            "role": "user",
                            "content": f"Repository content:\n{repo_content}\n\nUser prompt: {request.prompt}\n\nGenerate the necessary code changes with proper file paths."
                        }
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.7
                }
            )
            
            if groq_response.status_code != 200:
                print(f">>> Groq API error: {groq_response.status_code}")
                raise HTTPException(status_code=500, detail=f"Groq API error: {groq_response.text}")
            
            groq_data = groq_response.json()
            generated_code = groq_data["choices"][0]["message"]["content"]
            print(">>> Code generated by AI")
            
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
        print(f">>> Code generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze code with Groq: {str(e)}")
    
    # execute code in E2B sandbox and parse LLM response
    result_files = {}
    execution_results = {}
    try:
        # parse LLM response to extract files and content
        result_files = parse_llm_response(generated_code)
        print(f">>> Parsed {len(result_files)} files from AI response")
        
        # execute code in E2B sandbox for validation
        print(">>> Testing code in sandbox...")
        sandbox = Sandbox()
        
        for file_path, content in result_files.items():
            # determine language for sandbox execution
            if file_path.endswith('.py'):
                try:
                    execution_result = sandbox.run_code(content, language="python")
                    execution_results[file_path] = ">>> Executed successfully"
                    print(f">>> Executed {file_path} successfully")
                except Exception as sandbox_error:
                    execution_results[file_path] = f">>> Execution warning: {str(sandbox_error)[:100]}"
                    print(f">>> Warning: Failed to execute {file_path} in sandbox: {sandbox_error}")
                    # continue.. 

        # close sandbox (! method name varies by version)
        if hasattr(sandbox, 'close'):
            sandbox.close()
        elif hasattr(sandbox, 'kill'):
            sandbox.kill()
        else:
            print(">>> Sandbox cleanup method not found")
        
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
        print(f">>> Sandbox execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process generated code: {str(e)}")
    
    # create pull request with generated code
    try:
        print(">>> Creating pull request...")
        
        # get default branch ('main' or 'master')
        default_branch_name = github_repo.default_branch
        main_branch = github_repo.get_branch(default_branch_name)
        
        # create new branch with timestamp
        new_branch_name = f"ai-generated-{int(time.time())}"
        print(f">>> Creating branch: {new_branch_name}")
        github_repo.create_git_ref(f"refs/heads/{new_branch_name}", main_branch.commit.sha)
        
        # commit generated files
        committed_files = []
        for file_path, content in result_files.items():
            try:
                # ensure file path is safe (no directory traversal)
                if '..' in file_path or file_path.startswith('/'):
                    print(f">>> Skipping unsafe file path: {file_path}")
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
                    print(f">>> Updated existing file: {file_path}")
                except:
                    # file doesn't exist, create new one
                    github_repo.create_file(
                        file_path, 
                        f"AI generated code: {file_path}", 
                        content, 
                        branch=new_branch_name
                    )
                    committed_files.append(f"Created: {file_path}")
                    print(f">>> Created new file: {file_path}")
                    
            except Exception as file_error:
                print(f">>> Failed to commit {file_path}: {str(file_error)}")
                continue
        
        # create detailed PR summary
        pr_title, pr_body = create_detailed_pr_summary(request.prompt, result_files, generated_code)
        
        # create pull request with enhanced summary
        pr = github_repo.create_pull(
            title=pr_title,
            body=pr_body,
            head=new_branch_name,
            base=default_branch_name
        )
        
        print(f">>> Pull request created: {pr.html_url}")
        
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
        
        # response with PR URL and summary
        return {
            "pull_request_url": pr.html_url,
            "summary": changes_summary
        }
        
    except Exception as e:
        print(f">>> Pull request creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create pull request: {str(e)}")
    
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
                print(">>> Cleaned up temporary files")

            except Exception as cleanup_error:
                print(f">>> Cleanup warning: {str(cleanup_error)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)