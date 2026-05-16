from langchain_core.messages import AIMessage, HumanMessage

from src.rag.llm import count_message_tokens, count_tokens


def test_turkish_token_counter_is_not_char_division(capsys):
    samples = [
        "Merhaba, bugün nasılsın?",
        "Bu belgede yapay zeka destekli arama mimarisi açıklanıyor.",
        "Türkçe karakterler token sayımını doğrudan etkiler.",
        "Kullanıcının sorusunu yanıtlamak için bağlam parçaları seçilir.",
        "Çok modlu RAG sistemi görüntü, metin ve web kaynaklarını birleştirir.",
    ]

    for text in samples:
        approx = len(text) / 4
        tokens = count_tokens(text)
        print(f"{tokens=} chars_div_4={approx:.2f} text={text}")
        assert tokens > 0
        assert abs(tokens - approx) >= 1

    assert "tokens=" in capsys.readouterr().out


def test_count_message_tokens_reads_langchain_content():
    messages = [HumanMessage(content="Merhaba"), AIMessage(content="Size nasıl yardımcı olabilirim?")]

    assert count_message_tokens(messages) == count_tokens("Merhaba") + count_tokens("Size nasıl yardımcı olabilirim?")
