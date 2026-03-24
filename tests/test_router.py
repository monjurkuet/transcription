from audio_transcript.services.router import ProviderRouter


def test_provider_router_round_robins_remote_order():
    router = ProviderRouter(["groq", "mistral"])

    assert router.select_remote_order({"groq": True, "mistral": True}) == ["groq", "mistral"]
    assert router.select_remote_order({"groq": True, "mistral": True}) == ["mistral", "groq"]
    assert router.select_remote_order({"groq": True, "mistral": False}) == ["groq"]
