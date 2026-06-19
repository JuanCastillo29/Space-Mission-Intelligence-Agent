"""Tests for prompt construction functions."""

from scripts.generation.prompt import (
    build_routing_prompt,
    build_system_prompt,
    build_user_prompt,
)


class TestBuildSystemPrompt:
    def test_contains_citation_rules(self):
        prompt = build_system_prompt()
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "CITATION RULES" in prompt

    def test_contains_response_format(self):
        prompt = build_system_prompt()
        assert "## Sources" in prompt

    def test_contains_insufficient_context_handling(self):
        prompt = build_system_prompt()
        assert (
            "don't have sufficient information" in prompt.lower()
            or "I don't have sufficient" in prompt
        )


class TestBuildUserPrompt:
    def test_includes_context_and_query(self):
        prompt = build_user_prompt("What is Rosetta?", "Context about Rosetta")
        assert "Context about Rosetta" in prompt
        assert "What is Rosetta?" in prompt

    def test_has_section_headers(self):
        prompt = build_user_prompt("query", "context")
        assert "## Source Documents" in prompt
        assert "## Question" in prompt


class TestBuildRoutingPrompt:
    def test_contains_query(self):
        prompt = build_routing_prompt("What is the ISS orbit?")
        assert "What is the ISS orbit?" in prompt

    def test_requests_json_output(self):
        prompt = build_routing_prompt("test")
        assert "JSON" in prompt

    def test_contains_all_categories(self):
        prompt = build_routing_prompt("test")
        assert "retrieval" in prompt
        assert "structured" in prompt
        assert "hybrid" in prompt

    def test_schema_has_query_type_field(self):
        prompt = build_routing_prompt("test")
        assert "query_type" in prompt
