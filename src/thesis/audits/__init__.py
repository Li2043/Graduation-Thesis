"""Scripted base-outcome audits (Stage 3A)."""

from thesis.audits.base_outcome_audit import run_full_audit
from thesis.audits.audit_scenarios import build_matched_blocks

__all__ = ["run_full_audit", "build_matched_blocks"]
