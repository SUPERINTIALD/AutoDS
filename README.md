# AutoDS

## Table of Contents
1. [Introduction](#introduction)
2. [Setup Instructions](#setup-instructions)
   - [WebUI Setup](#webui-setup)
   - [DeepSeek Setup](#deepseek-setup)
3. [How to Use](#how-to-use)
---

## Introduction
**AutoDS** is a project focused on creating intelligent agents powered by LLMs. This includes exploring cutting-edge models like DeepSeek and integrating tools like WebUI for seamless interaction and deployment.

---

## Setup Instructions


### DeepSeek Setup

1. **Download DeepSeek Model:**
   - Visit the [DeepSeek R1 Library](https://ollama.com/library/deepseek-r1).

2. **Download the model to your local machine. 14B is fast, and 9 GB of storage**
    - ollama run deepseek-r1:14b
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
   python3 webui.py
   ```
   - Open your browser and navigate to the provided URL (usually `http://localhost:8080`).

---