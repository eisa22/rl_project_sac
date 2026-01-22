"""
MTMH-SAC (Multi-Task Multi-Head Soft Actor-Critic) for Stable-Baselines3.

This package provides:
- MHSACAgentSB3: Multi-head SAC agent for multi-task learning
- SACAgentSB3: Standard SAC agent
- Curriculum learning utilities
- Evaluation utilities with IQM and confidence intervals
"""

from .agent import SACAgentSB3, SACAgentSB3Config
from .curriculum import CurriculumTracker, MetaWorldCurriculumEnv, make_curriculum_vec_env
from .multiheadAgent import (
    MHSACAgentSB3, 
    MHSACAgentSB3Config, 
    MultiHeadActor, 
    MultiHeadCritic,
    MultiHeadLinear,
    MultiTaskTemperature,
    AsymSACPolicy,
    MTMHSACLoggingCallback,
)
from .evaluation import (
    interquartile_mean,
    bootstrap_ci,
    stratified_bootstrap_ci,
    TaskEvalResult,
    MultiTaskEvalResult,
    MultiSeedEvalResult,
    evaluate_single_task,
    evaluate_multi_task,
    evaluate_multi_seed,
    OnlineEvaluationTracker,
)

__all__ = [
    # Agents
    "SACAgentSB3",
    "SACAgentSB3Config", 
    "MHSACAgentSB3",
    "MHSACAgentSB3Config",
    # Networks
    "MultiHeadActor",
    "MultiHeadCritic",
    "MultiHeadLinear",
    "MultiTaskTemperature",
    "AsymSACPolicy",
    # Curriculum
    "CurriculumTracker",
    "MetaWorldCurriculumEnv",
    "make_curriculum_vec_env",
    # Evaluation
    "interquartile_mean",
    "bootstrap_ci",
    "stratified_bootstrap_ci",
    "TaskEvalResult",
    "MultiTaskEvalResult",
    "MultiSeedEvalResult",
    "evaluate_single_task",
    "evaluate_multi_task",
    "evaluate_multi_seed",
    "OnlineEvaluationTracker",
    # Callbacks
    "MTMHSACLoggingCallback",
]
