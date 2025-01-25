# AutoDS

## Table of Contents
1. [Introduction](#introduction)
2. [Setup Instructions](#setup-instructions)
   - [WebUI Setup](#webui-setup)
   - [DeepSeek Setup](#deepseek-setup)
---

## Introduction
**AutoDS** is a project focused on creating intelligent agents powered by LLMs. This includes exploring cutting-edge models like DeepSeek and integrating tools like WebUI for seamless interaction and deployment.
AutoDS leverages **LangChain/Graph** to create a multi-agent system that utilizes tools for specific tasks. It improves the **Data-to-Paper** workflow by integrating it into a single, magnetic system. The ultimate goal is for AutoDS to enable AI to autonomously decide and optimize the workflow.




---

## Setup Instructions


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