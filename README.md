# AutoDS

This README is for the first prototype and will later be replaced with the full README.

---

# LangChain Simple Conversational Demo

A demonstration of LangChain for building conversational AI applications using OpenAI's GPT models. This repository includes reference code for initializing an agent, using memory, and integrating tools for natural language processing tasks.

---

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
