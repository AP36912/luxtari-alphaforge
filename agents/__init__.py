"""
Supervised research agent layer (Phase 12).

Agents orchestrate existing ``quant_ai`` modules with structured dataclass I/O.
The supervisor runs a fixed, deterministic pipeline suitable for tests and future LLM hooks.
"""

from agents.backtest_agent import BacktestAgentInput, run_backtest_agent
from agents.data_agent import DataAgentInput, run_data_agent
from agents.feature_agent import FeatureAgentInput, run_feature_agent
from agents.modeling_agent import ModelingAgentInput, run_modeling_agent
from agents.research_writer_agent import ResearchWriterInput, run_research_writer_agent
from agents.supervisor import build_experiment_plan, run_supervised_research
from agents.types import ResearchObjective, SupervisorResult
from agents.validation_agent import ValidationAgentInput, run_validation_agent

__all__ = [
    "BacktestAgentInput",
    "DataAgentInput",
    "FeatureAgentInput",
    "ModelingAgentInput",
    "ResearchObjective",
    "ResearchWriterInput",
    "SupervisorResult",
    "ValidationAgentInput",
    "build_experiment_plan",
    "run_backtest_agent",
    "run_data_agent",
    "run_feature_agent",
    "run_modeling_agent",
    "run_research_writer_agent",
    "run_supervised_research",
    "run_validation_agent",
]
