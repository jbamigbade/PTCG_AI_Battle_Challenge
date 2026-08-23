from __future__ import annotations

from .base_agent import BaseAgent
from .configs import PolicyConfig
from .policies import (
    AggressivePolicy,
    BasePolicy,
    DefensivePolicy,
    GreedyPolicy,
    HeuristicPolicy,
    RandomPolicy,
    SearchPolicy,
)


class PolicyFactory:

    _registry = {
        "random": RandomPolicy,
        "greedy": GreedyPolicy,
        "heuristic": HeuristicPolicy,
        "search": SearchPolicy,
        "aggressive": AggressivePolicy,
        "defensive": DefensivePolicy,
    }

    @classmethod
    def create(
        cls,
        config: PolicyConfig,
    ) -> BasePolicy:

        try:
            policy_class = cls._registry[
                config.policy_type
            ]
        except KeyError as error:
            raise ValueError(
                f"Unknown policy "
                f"{config.policy_type!r}"
            ) from error

        return policy_class(config)

    @classmethod
    def available_policies(
        cls,
    ) -> list[str]:

        return sorted(cls._registry)


class AgentFactory:

    @staticmethod
    def create(
        name: str,
        policy_type: str,
        **kwargs,
    ) -> BaseAgent:

        config = PolicyConfig(
            name=f"{policy_type.title()}Policy",
            policy_type=policy_type,
            **kwargs,
        )

        return BaseAgent(
            name=name,
            policy=PolicyFactory.create(config),
        )
