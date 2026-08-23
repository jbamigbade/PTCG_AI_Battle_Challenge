
# Pokémon TCG Final PPO Agent

## Entry Point

from src.agents.final_ppo_agent import build_final_agent

agent = build_final_agent(
    checkpoint_path="models/ppo_training_loop.pkl"
)

This package contains the production PPO policy engine,
battle agent wrapper, and trained checkpoint.

The current checkpoint is a bootstrap validation checkpoint.
