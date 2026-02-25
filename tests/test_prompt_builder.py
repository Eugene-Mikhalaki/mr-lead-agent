"""Tests for prompt builder."""

from __future__ import annotations

from mr_lead_agent.prompt_builder import build_prompt


class TestBuildPrompt:
    def test_prompt_contains_all_sections(self, minimal_config, sample_mr) -> None:
        prompt = build_prompt(sample_mr, [], minimal_config)
        assert "ROLE & POLICY" in prompt
        assert "MR METADATA" in prompt
        assert "DIFF" in prompt
        assert "OUTPUT CONTRACT" in prompt

    def test_prompt_contains_mr_title(self, minimal_config, sample_mr) -> None:
        prompt = build_prompt(sample_mr, [], minimal_config)
        assert sample_mr.title in prompt

    def test_prompt_contains_diff(self, minimal_config, sample_mr) -> None:
        prompt = build_prompt(sample_mr, [], minimal_config)
        assert sample_mr.diff in prompt

    def test_context_fragments_included(self, minimal_config, sample_mr, sample_fragment) -> None:
        prompt = build_prompt(sample_mr, [sample_fragment], minimal_config)
        assert "RETRIEVED CONTEXT" in prompt
        assert sample_fragment.file_path in prompt

    def test_large_diff_triggers_summary_mode(self, minimal_config, sample_mr) -> None:
        big_diff = "\n".join(["+line"] * 4000)
        large_mr = sample_mr.model_copy(update={"diff": big_diff})
        prompt = build_prompt(large_mr, [], minimal_config)
        assert "large" in prompt.lower() or "summary" in prompt.lower()

    def test_output_schema_in_prompt(self, minimal_config, sample_mr) -> None:
        prompt = build_prompt(sample_mr, [], minimal_config)
        assert "blockers" in prompt
        assert "summary" in prompt
        assert "key_risks" in prompt
