# Backspacely
`AI Coding Agent: A full pull-request automation agent`

## Problem Definition
- **Poor P2C Flow:** AI tools only generate snippets instead of full pull requests

- **Repetition:** Repetitive manual tasks from creating branches to making pull requests

- **Context Switching:** Analyzing unfamiliar codebases takes time

## Key Features
- **Prompt Understandings:** Returns PR link and change summary

- **Secure Repo Cloning:** Uses E2B sandboxing to safely clone and run code
  
- **Automated Git workflow:** Creates branch, commits code, opens pull request

- **Intelligent Code Editing:** Generates and applies changes using Meta’s ollama via Groq Cloud API

- **Real-time Streaming:** SSE updates shows every step; clone → edit → PR

- **Clear Output:** Returns PR link and change summary

## Tech Stack
- **Languages:** Python
- **APIs & Services:** E2B Sandbox, Groq API, Ollama
- **Libraries & Frameworks:** FastAPI
- **Tools:** Git

## Demo & Pitch
Click image below ⬇️

[![Backspacely Demo](https://img.youtube.com/vi/M4tsB48rT8g/0.jpg)](https://www.youtube.com/watch?v=M4tsB48rT8g)

## Limitations
- No access to private GitHub repositories; authentication is limited to public data only

- No telemetry or observability layer; agent reasoning and decision paths are not externally visualized

- Single-agent only; the system does not coordinate or orchestrate multiple agents


## License
- Distributed under the ***BSD 3-Clause License***. See **LICENSE** for more information.



  
