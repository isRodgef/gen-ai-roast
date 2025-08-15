# Gen AI Roast Battle

A fun project that pits two AI agents against each other in a roast battle using Ollama and the llama2-uncensored:7b model. The agents maintain memory of the conversation, allowing for more contextual and entertaining exchanges.

## Features

- Two AI agents with distinct personalities roasting each other
- Conversation memory that allows agents to reference previous exchanges
- Configurable number of rounds and delay between responses
- Uses the llama2-uncensored:7b model via Ollama

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
  - Default connection: 127.0.0.1:11434 (standard port)
  - For this environment, use: 127.0.0.1:1143
- llama2-uncensored:7b model pulled in Ollama (will be pulled automatically if not present)

## Setup

1. Make sure Ollama is installed. If not, follow the instructions at [ollama.ai](https://ollama.ai/)

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running in a separate terminal:

```bash
ollama serve
```

4. Run the roast battle:

```bash
python roast_battle.py
```

## Customization

You can customize the roast battle with command-line arguments:

```bash
python roast_battle.py --rounds 5 --delay 2.0 --host localhost --port 11434
```

- `--rounds`: Number of rounds in the battle (default: 3)
- `--delay`: Delay between responses in seconds (default: 1.0)
- `--host`: Ollama API host (default: from OLLAMA_HOST env var or localhost)
- `--port`: Ollama API port (default: from OLLAMA_PORT env var or 11434)
- `--no-memory`: Disable loading previous memories
- `--no-save`: Don't save transcript or memories
- `--list-battles`: List past battles and exit
- `--session`: Continue a specific session by ID

## How It Works

The application creates two AI agents with different personalities and puts them in a roast battle. Each agent maintains memory of the conversation, allowing them to reference previous exchanges and build upon them for more coherent and entertaining roasts.

## Disclaimer

This project uses the llama2-uncensored:7b model, which may generate content that some users find offensive. The roasts are meant to be humorous but may contain adult language and themes. Use at your own discretion.