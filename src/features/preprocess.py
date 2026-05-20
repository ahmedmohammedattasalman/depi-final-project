def preprocess_text(text):
    """تطبيع النصوص للاستخدام في embeddings إذا لزم الأمر"""
    if not isinstance(text, str):
        return ""
    return text.strip().lower()
