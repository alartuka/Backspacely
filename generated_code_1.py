# Generate hello world HTML page
result_files = {
    "index.html": """
<!DOCTYPE html>
<html>
<head>
    <title>Hello World</title>
</head>
<body>
    <h1>Hello, World!</h1>
</body>
</html>
""",
}

# Create pull request with generated code
pr_title, pr_body = create_detailed_pr_summary(request.prompt, result_files, result_files["index.html"])
pr = github_repo.create_pull(
    title=pr_title,
    body=pr_body,
    head=new_branch_name,
    base=default_branch_name
)

print(f">>> Pull request created: {pr.html_url}")

# Create summary of changes
file_types = {}
total_lines = 0
for file_path, content in result_files.items():
    ext = file_path.split('.')[-1] if '.' in file_path else 'unknown'
    file_types[ext] = file_types.get(ext, 0) + 1
    total_lines += len(content.split('\n'))

file_summary = ", ".join([f"{count} {ext} file{'s' if count > 1 else ''}" 
                         for ext, count in file_types.items()])

changes_summary = f"Generated {len(result_files)} files ({file_summary}) with {total_lines} total lines of code based on prompt: '{request.prompt}'"

# Response with PR URL and summary
return {
    "pull_request_url": pr.html_url,
    "summary": changes_summary
}