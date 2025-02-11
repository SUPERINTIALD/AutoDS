# AutoDS

## Table of Contents
1. [Introduction](#introduction)
2. [Setup Instructions](#Setup)
   - [WebUI Setup](#webui-setup)
   - [DeepSeek Setup](#deepseek-setup)
---

## Introduction
**AutoDS** is a project focused on creating intelligent agents powered by LLMs. This includes exploring cutting-edge models like DeepSeek and integrating tools like WebUI for seamless interaction and deployment.
AutoDS leverages **LangChain/Graph** to create a multi-agent system that utilizes tools for specific tasks. It improves the **Data-to-Paper** workflow by integrating it into a single, magnetic system. The ultimate goal is for AutoDS to enable AI to autonomously decide and optimize the workflow.




---

## Setup Instructions


## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AutoDS.git
cd AutoDS
```

### 2. Set Up a Virtual Environment
- **Windows**:
  ```bash
  python -m venv venv
  venv\\Scripts\\activate
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
Install the required dependencies using `uv pip` (or `pip` if `uv` is not available):
```bash
uv pip install -r requirements.txt
```
*Note: If `uv` is not installed, use `pip install -r requirements.txt`.*

### 4. Create a `.env` File
Create a `.env` file in the root directory and add your OpenAI API key:
```plaintext
OPENAI_API_KEY=your_api_key_here
```

### 5. Run the Application
Execute the main script to start the conversational agent:
```bash
python main.py
```

---

## Command Reference

### Update `requirements.txt`
If you’ve added a new dependency, update the `requirements.txt` file:
```bash
pip freeze > requirements.txt
```

### Useful Git Commands
- **Check Status**:
  ```bash
  git status
  ```
- **Add Changes**:
  ```bash
  git add .
  ```
- **Commit Changes**:
  ```bash
  git commit -m "Your commit message"
  ```
- **Push Changes**:
  ```bash
  git push origin main
  ```
---

### DeepSeek Setup

1. **Download DeepSeek Model:**
   - Visit the [DeepSeek R1 Library](https://ollama.com/library/deepseek-r1).

2. **Download the model to your local machine. 14B is fast, and 9 GB of storage**
    ```bash
    ollama run deepseek-r1:14b
    ```
---


### WebUI Setup

1. **Install Required Dependencies:**
   Ensure you have `pip` installed. Run the following command:
   ```bash
   uv pip install open-webui
   ```

2. **Start the WebUI:**
   Once the dependencies are installed, start the WebUI server:
   ```bash
    open-webui serve   
    ```
   - Open your browser and navigate to the provided URL (usually `http://localhost:8080`).

---


# LangChain Simple Conversational Demo

A demonstration of LangChain for building conversational AI applications using OpenAI's GPT models. This repository includes reference code for initializing an agent, using memory, and integrating tools for natural language processing tasks.

---
