from text2sql_agent.ui_questions import EXAMPLE_QUESTIONS


def test_example_questions_cover_major_business_areas():
    assert len(EXAMPLE_QUESTIONS) >= 12
    labels = " ".join(EXAMPLE_QUESTIONS).lower()
    for topic in ("delivery", "revenue", "payment", "customer", "seller", "review"):
        assert topic in labels


def test_example_questions_have_unique_nonempty_prompts():
    prompts = list(EXAMPLE_QUESTIONS.values())

    assert all(prompt.strip().endswith(("?", ".")) for prompt in prompts)
    assert len(prompts) == len(set(prompts))
