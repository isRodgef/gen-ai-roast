#!/usr/bin/env python3
"""
Roast Battle - Two Ollama agents roasting each other using llama2-uncensored:7b
with persistent conversation memory.
"""

import os
import time
import json
import random
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import ollama

# Constants
MODEL_NAME = "llama2-uncensored:7b"
MAX_MEMORY = 10  # Maximum number of exchanges to remember in short-term memory
MEMORY_DIR = Path("memory")  # Directory to store persistent memories

class Agent:
    def __init__(self, name: str, personality: str, session_id: str = None):
        """Initialize an agent with a name and personality."""
        self.name = name
        self.personality = personality
        self.session_id = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.memory: List[Dict[str, str]] = []
        
        # Create memory directory if it doesn't exist
        MEMORY_DIR.mkdir(exist_ok=True)
        
        # Load previous memories if they exist
        self.load_memories()
    
    def remember(self, speaker: str, message: str):
        """Add a message to the agent's memory."""
        timestamp = datetime.datetime.now().isoformat()
        memory_entry = {
            "speaker": speaker, 
            "message": message,
            "timestamp": timestamp
        }
        
        self.memory.append(memory_entry)
        # Keep working memory within limits
        if len(self.memory) > MAX_MEMORY:
            self.memory.pop(0)
        
        # Save to persistent storage
        self.save_memory(memory_entry)
    
    def format_memory(self, include_long_term: bool = True) -> str:
        """Format the memory for inclusion in the prompt."""
        if not self.memory:
            return "No previous conversation."
        
        memory_text = "Recent conversation:\n"
        for entry in self.memory:
            memory_text += f"{entry['speaker']}: {entry['message']}\n"
        
        # Add long-term memories if requested
        if include_long_term:
            long_term_memories = self.get_relevant_long_term_memories()
            if long_term_memories:
                memory_text += "\nMemories from previous conversations:\n"
                for memory in long_term_memories:
                    memory_text += f"{memory['speaker']}: {memory['message']}\n"
        
        return memory_text
    
    def save_memory(self, memory_entry: Dict[str, str]):
        """Save a memory entry to persistent storage."""
        agent_dir = MEMORY_DIR / self.name
        agent_dir.mkdir(exist_ok=True)
        
        # Create a file for this session if it doesn't exist
        session_file = agent_dir / f"{self.session_id}.jsonl"
        
        # Append the memory to the file
        with open(session_file, "a") as f:
            f.write(json.dumps(memory_entry) + "\n")
    
    def load_memories(self):
        """Load memories from persistent storage."""
        agent_dir = MEMORY_DIR / self.name
        if not agent_dir.exists():
            return
        
        # Load the most recent session if it exists (excluding current session)
        session_files = list(agent_dir.glob("*.jsonl"))
        if not session_files:
            return
        
        # Sort by modification time (most recent first)
        session_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Load the most recent session that's not the current one
        for session_file in session_files:
            if session_file.stem != self.session_id:
                try:
                    with open(session_file, "r") as f:
                        # Load only the most recent MAX_MEMORY entries
                        lines = f.readlines()[-MAX_MEMORY:]
                        for line in lines:
                            self.memory.append(json.loads(line.strip()))
                        break
                except (json.JSONDecodeError, FileNotFoundError):
                    continue
    
    def get_relevant_long_term_memories(self, limit: int = 5) -> List[Dict[str, str]]:
        """Get relevant memories from past sessions."""
        agent_dir = MEMORY_DIR / self.name
        if not agent_dir.exists():
            return []
        
        # Collect memories from all sessions except current
        all_memories = []
        session_files = list(agent_dir.glob("*.jsonl"))
        
        for session_file in session_files:
            if session_file.stem != self.session_id:
                try:
                    with open(session_file, "r") as f:
                        for line in f:
                            try:
                                memory = json.loads(line.strip())
                                all_memories.append(memory)
                            except json.JSONDecodeError:
                                continue
                except FileNotFoundError:
                    continue
        
        # For now, just return the most recent ones
        # In a more advanced implementation, we could use semantic search
        # to find the most relevant memories based on the current conversation
        return sorted(all_memories, 
                      key=lambda x: x.get("timestamp", ""), 
                      reverse=True)[:limit]
    
    def generate_roast(self, opponent_name: str) -> str:
        """Generate a roast directed at the opponent."""
        memory_context = self.format_memory()
        
        prompt = f"""
{memory_context}

You are {self.name}, {self.personality}. 
You are in a roast battle against {opponent_name}.
Your goal is to come up with the most creative, hilarious, and savage roast.
Be ruthless but clever. Focus on being funny rather than just mean.
Keep your response to one or two paragraphs maximum.

Now roast {opponent_name}:
        """
        
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response['message']['content'].strip()


class RoastBattle:
    def __init__(self, rounds: int = 5, delay: float = 2.0, session_id: str = None):
        """Initialize a roast battle with a specified number of rounds."""
        self.rounds = rounds
        self.delay = delay
        self.session_id = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create our two roasters with distinct personalities
        self.agent1 = Agent(
            "RoastMaster", 
            "a confident, quick-witted comedian known for brutal takedowns",
            session_id=self.session_id
        )
        self.agent2 = Agent(
            "BurnBot", 
            "a sarcastic, observant AI with a talent for finding insecurities",
            session_id=self.session_id
        )
    
    def battle(self, save_transcript: bool = True):
        """Run the roast battle for the specified number of rounds."""
        print("\n🔥 ROAST BATTLE BEGINS 🔥\n")
        print(f"{self.agent1.name} VS {self.agent2.name}")
        print("=" * 50)
        
        # Check if agents have memories from previous sessions
        if any(self.agent1.get_relevant_long_term_memories()) or any(self.agent2.get_relevant_long_term_memories()):
            print("\n💭 The agents remember their previous encounters...\n")
        
        # First round starts with agent1
        current_speaker = self.agent1
        opponent = self.agent2
        
        # Keep track of all exchanges for the transcript
        transcript = []
        
        for round_num in range(1, self.rounds + 1):
            print(f"\nROUND {round_num}")
            print("-" * 20)
            
            # Each agent gets a turn in each round
            for _ in range(2):
                print(f"\n{current_speaker.name}'s turn:")
                roast = current_speaker.generate_roast(opponent.name)
                print(f"{current_speaker.name}: {roast}")
                
                # Record this exchange
                exchange = {
                    "round": round_num,
                    "speaker": current_speaker.name,
                    "target": opponent.name,
                    "message": roast,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                transcript.append(exchange)
                
                # Both agents remember this exchange
                current_speaker.remember(current_speaker.name, roast)
                opponent.remember(current_speaker.name, roast)
                
                # Switch roles
                current_speaker, opponent = opponent, current_speaker
                
                # Add a delay between turns for readability
                time.sleep(self.delay)
        
        print("\n🏆 ROAST BATTLE COMPLETE 🏆\n")
        
        # Save the transcript if requested
        if save_transcript:
            self._save_transcript(transcript)
    
    def _save_transcript(self, transcript: List[Dict[str, Any]]):
        """Save the battle transcript to a file."""
        # Create transcripts directory if it doesn't exist
        transcript_dir = Path("transcripts")
        transcript_dir.mkdir(exist_ok=True)
        
        # Save the transcript as JSON
        transcript_file = transcript_dir / f"battle_{self.session_id}.json"
        with open(transcript_file, "w") as f:
            json.dump({
                "session_id": self.session_id,
                "agent1": self.agent1.name,
                "agent2": self.agent2.name,
                "rounds": self.rounds,
                "exchanges": transcript
            }, f, indent=2)
        
        print(f"Transcript saved to {transcript_file}")


def list_past_battles():
    """List all past battles with their session IDs."""
    transcript_dir = Path("transcripts")
    if not transcript_dir.exists() or not list(transcript_dir.glob("*.json")):
        print("No past battles found.")
        return
    
    print("\nPast Battles:\n")
    for transcript_file in sorted(transcript_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(transcript_file, "r") as f:
                data = json.load(f)
                timestamp = data.get("session_id", "Unknown")
                agent1 = data.get("agent1", "Unknown")
                agent2 = data.get("agent2", "Unknown")
                rounds = data.get("rounds", 0)
                print(f"- Session: {timestamp} | {agent1} vs {agent2} | {rounds} rounds")
        except (json.JSONDecodeError, FileNotFoundError):
            continue


def main():
    parser = argparse.ArgumentParser(description="AI Roast Battle using Ollama")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds (default: 3)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between responses in seconds (default: 1.0)")
    parser.add_argument("--no-memory", action="store_true", help="Disable loading previous memories")
    parser.add_argument("--no-save", action="store_true", help="Don't save transcript or memories")
    parser.add_argument("--list-battles", action="store_true", help="List past battles and exit")
    parser.add_argument("--session", type=str, help="Continue a specific session by ID")
    parser.add_argument("--host", type=str, help="Ollama API host (default: from OLLAMA_HOST env var or localhost)")
    parser.add_argument("--port", type=int, help="Ollama API port (default: from OLLAMA_PORT env var or 11434)")
    args = parser.parse_args()
    
    # Set Ollama host and port from environment variables or command line
    if args.host:
        os.environ["OLLAMA_HOST"] = args.host
    if args.port:
        os.environ["OLLAMA_PORT"] = str(args.port)
        
    # Print connection information
    host = os.environ.get("OLLAMA_HOST", "localhost")
    port = os.environ.get("OLLAMA_PORT", "11434")
    print(f"Connecting to Ollama server at {host}:{port}...")
    
    # List past battles if requested
    if args.list_battles:
        list_past_battles()
        return
    
    try:
        # Check if Ollama is available and the model is installed
        try:
            models = ollama.list()
            # Check if the model exists in the response
            model_names = [model.get('name') for model in models.get('models', [])] 
            model_installed = MODEL_NAME in model_names
            
            if not model_installed:
                print(f"Model {MODEL_NAME} not found. Pulling it now...")
                ollama.pull(MODEL_NAME)
                print(f"Model {MODEL_NAME} successfully pulled.")
        except Exception as e:
            print(f"Error connecting to Ollama server: {e}")
            print("\nPlease make sure Ollama is installed and running.")
            print("Run 'ollama serve' in a separate terminal, then try again.")
            return
        
        # If no-memory is set, rename the memory directory temporarily
        if args.no_memory and MEMORY_DIR.exists():
            temp_memory_dir = MEMORY_DIR.with_name(f"{MEMORY_DIR.name}_backup")
            MEMORY_DIR.rename(temp_memory_dir)
            MEMORY_DIR.mkdir(exist_ok=True)
        
        # Start the roast battle
        session_id = args.session if args.session else None
        battle = RoastBattle(rounds=args.rounds, delay=args.delay, session_id=session_id)
        battle.battle(save_transcript=not args.no_save)
        
        # Restore memory directory if it was temporarily renamed
        if args.no_memory and MEMORY_DIR.exists() and Path(f"{MEMORY_DIR.name}_backup").exists():
            MEMORY_DIR.rmdir()  # Should be empty if we just created it
            Path(f"{MEMORY_DIR.name}_backup").rename(MEMORY_DIR)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running and accessible.")
        print("Run 'ollama serve' in a separate terminal if needed.")


if __name__ == "__main__":
    main()
