# file_utils.py
"""
Содержит функции для загрузки конфигурации из файлов (сущности, исключения).
ИЗМЕНЕНО: Функции сделаны асинхронными с использованием aiofiles.
"""
import logging
import asyncio
import aiofiles # <-- Добавлено
from config import ENTITY_PLACEHOLDERS # Импортируем для fallback в load_entities

async def load_entities_to_process(filename: str) -> list[str]: # <-- async def
    """Загружает список сущностей для обработки из файла (асинхронно)."""
    entities = []
    try:
        # Используем async with и aiofiles.open
        async with aiofiles.open(filename, mode='r', encoding='utf-8') as f:
            # Используем async for
            async for line in f:
                cleaned_line = line.split('#')[0].strip()
                if cleaned_line:
                    entities.append(cleaned_line)
        if not entities:
             logging.warning(f"Файл сущностей '{filename}' пуст или содержит только комментарии.")
        logging.info(f"Загружены сущности для обработки из '{filename}': {entities}")
        return entities
    except FileNotFoundError:
        logging.warning(f"Файл со списком сущностей '{filename}' не найден.")
        default_entities = list(ENTITY_PLACEHOLDERS.keys())
        if "DEFAULT" in default_entities:
            default_entities.remove("DEFAULT")
        logging.warning(f"Используется стандартный список сущностей: {default_entities}")
        return default_entities
    except Exception as e:
        logging.error(f"Ошибка при чтении файла сущностей '{filename}': {e}")
        return []

async def load_exceptions(filename: str) -> set[str]: # <-- async def
    """Загружает список исключений из файла (в нижнем регистре, асинхронно)."""
    exceptions = set()
    try:
        # Используем async with и aiofiles.open
        async with aiofiles.open(filename, mode='r', encoding='utf-8') as f:
            # Используем async for
            async for line in f:
                cleaned_line = line.split('#')[0].strip()
                if cleaned_line:
                    exceptions.add(cleaned_line.lower())
        logging.info(f"Загружены исключения из '{filename}' ({len(exceptions)} шт.): {exceptions if len(exceptions) < 10 else str(list(exceptions)[:10])+'...'}")
    except FileNotFoundError:
        logging.info(f"Файл исключений '{filename}' не найден. Исключения не используются.")
    except Exception as e:
        logging.error(f"Ошибка при чтении файла исключений '{filename}': {e}")
    return exceptions