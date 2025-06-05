# config.py
"""
Содержит основные настройки и константы приложения.
ИЗМЕНЕНО: Добавлены пороги и множители для настройки логики score.
"""

# --- Имена файлов ---
INPUT_FILENAME = "input.txt"
OUTPUT_FILENAME = "output.txt"
ENTITIES_FILENAME = "entities.txt"
EXCEPTIONS_FILENAME = "exceptions.txt"

# --- Настройки языка и моделей ---
LANGUAGE_CODE = "ru"
SPACY_MODEL_RU = "ru_core_news_lg"
SPACY_MODEL_EN = "en_core_web_sm"

# --- Настройки Presidio ---
DEFAULT_SCORE_THRESHOLD = 0.55 # Минимальный score для учета результата анализатором

# --- НОВЫЕ Настройки логики Score и Фильтрации ---
# Порог score для определения "якорного" результата (Stanza или высокий score)
ANCHOR_SCORE_THRESHOLD = 0.95
# Порог score, ниже которого к NER-результатам применяется расширенная проверка по NER_FALSE_POSITIVE_FILTER
NER_FILTER_LOW_SCORE_THRESHOLD = 0.75
# Множитель для понижения score подозрительных NER результатов (например, начинающихся со строчной буквы)
NER_LOW_CONFIDENCE_SCORE_MULTIPLIER = 0.5
# Базовый score, присваиваемый результатам Natasha NER (т.к. Natasha сама score не дает)
NATASHA_DEFAULT_SCORE = 0.85
# -------------------------------------------------

# --- Настройки Оборудования ---
# Установите True для попытки использования GPU (CUDA), False для использования CPU
USE_GPU = False

# --- Плейсхолдеры для замены ---
ENTITY_PLACEHOLDERS = {
    "PERSON": "<ФИО>",
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<ТЕЛЕФОН>",
    "LOCATION": "<МЕСТО>",
    "DATE_TIME": "<ДАТА_ВРЕМЯ>",
    "NRP": "<НАЦИОНАЛЬНОСТЬ>",
    "URL": "<URL>",
    "IP_ADDRESS": "<ДАННЫЕ УДАЛЕНЫ>",
    "CREDIT_CARD": "<ДАННЫЕ УДАЛЕНЫ>",
    "IBAN_CODE": "<IBAN>",
    "ORG": "<ОРГАНИЗАЦИЯ>",
    # Пользовательские
    "RU_QUOTED_DATE": "<ДАТА>",
    "RU_ADDRESS_PART": "<АДРЕС>",
    "RU_POSTAL_CODE": "<ИНДЕКС>",
    "RU_IDENTIFIER": "<ДАННЫЕ УДАЛЕНЫ>",
    "RU_CADASTRAL_NUMBER": "<КАДАСТР_НОМЕР>",
    # По умолчанию
    "DEFAULT": "<СКРЫТО>"
}