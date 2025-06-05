# text_utils.py
"""
Содержит утилиты для пост-обработки текста.
"""
import re
import logging

def post_process_text(text: str) -> str:
    """
    Выполняет пост-обработку текста: удаление лишних пробелов/табов
    и слияние плейсхолдеров, сохраняя переносы строк.
    """
    logging.debug("Выполнение пост-обработки текста...")
    # Сначала объединяем одинаковые соседние плейсхолдеры, разделенные только пробелами/табами/переносами
    processed_text = re.sub(r'(?P<tag><[\w_]+>)(?:\s*(?P=tag))+', r'\1', text)
    # Удаляем лишние пробелы и табы внутри строк
    processed_text = re.sub(r'[ \t]{2,}', ' ', processed_text)
    # Удаляем пробелы перед знаками препинания
    processed_text = re.sub(r'\s+([.,!?;:])', r'\1', processed_text)
    # Удаляем пробелы после открывающих скобок/кавычек
    processed_text = re.sub(r'([\(«"\'])(\s+)', r'\1', processed_text)
    # Удаляем пробелы перед закрывающими скобками/кавычками
    processed_text = re.sub(r'(\s+)([\)»"\'])', r'\2', processed_text)
    # Удаляем пробелы/табы в конце строк
    processed_text = re.sub(r'[ \t]+\n', '\n', processed_text)
    # Удаляем пробелы/табы в начале строк (после переноса)
    processed_text = re.sub(r'\n[ \t]+', '\n', processed_text)
    # Удаляем пробелы/табы в начале и конце всего текста
    processed_text = processed_text.strip()
    logging.debug("Пост-обработка завершена.")
    return processed_text