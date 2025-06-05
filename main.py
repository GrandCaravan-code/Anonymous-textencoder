# main.py
"""
Точка входа для скрипта анонимизации.
Отвечает за настройку логирования, проверку моделей,
загрузку конфигурации и запуск основного процесса анонимизации.
ИЗМЕНЕНО: Использует asyncio для запуска асинхронных операций.
"""
import logging
import sys
import os
import asyncio # <-- Добавлено

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
try:
    from logger_config import setup_logging
    setup_logging(level=logging.DEBUG, log_file="log.txt")
except ImportError as e:
    print(f"CRITICAL: Не удалось импортировать настройщик логирования: {e}", file=sys.stderr)
    exit(1)
except Exception as e:
    print(f"CRITICAL: Не удалось настроить логирование: {e}", file=sys.stderr)
    exit(1)
# --- КОНЕЦ НАСТРОЙКИ ЛОГИРОВАНИЯ ---

# --- ПОЛУЧАЕМ ЭКЗЕМПЛЯР ЛОГГЕРА ---
logger = logging.getLogger()
# --- КОНЕЦ ПОЛУЧЕНИЯ ЭКЗЕМПЛЯРА ---


# Импортируем остальные зависимости
try:
    import spacy
    import stanza
    from config import (
        INPUT_FILENAME, OUTPUT_FILENAME, ENTITIES_FILENAME, EXCEPTIONS_FILENAME,
        LANGUAGE_CODE, SPACY_MODEL_RU, SPACY_MODEL_EN,
        USE_GPU
    )
    # Импортируем асинхронные версии функций
    from file_utils import load_entities_to_process, load_exceptions # <-- Теперь это async функции
    from anonymizer_logic import anonymize_text_file # <-- Теперь это async функция
except ImportError as import_error:
    logger.critical(f"Ошибка импорта необходимой библиотеки: {import_error}")
    logger.critical("Убедитесь, что установлены 'spacy', 'stanza', 'presidio-analyzer', 'presidio-anonymizer', 'aiofiles'.") # <-- Добавлено aiofiles
    logger.critical("Пожалуйста, установите зависимости из файла requirements.txt: pip install -r requirements.txt")
    logger.critical("Убедитесь, что установлены модели spaCy и Stanza!")
    exit(1)

# --- Настройка использования GPU/CPU для spaCy (остается синхронной) ---
def setup_spacy_device():
    """Настраивает предпочтительное устройство для spaCy на основе config.USE_GPU."""
    if USE_GPU:
        logger.info("Попытка включить использование GPU для spaCy (если доступно).")
        try:
            gpu_activated = spacy.prefer_gpu()
            if gpu_activated:
                logger.info("spaCy успешно настроен для использования GPU.")
            else:
                logger.warning("GPU недоступен или не удалось активировать для spaCy. Будет использоваться CPU.")
                spacy.require_cpu()
                logger.info("spaCy настроен для использования CPU.")
        except Exception as e:
            logger.warning(f"Ошибка при вызове spacy.prefer_gpu(): {e}. Попытка использовать CPU.")
            try:
                 spacy.require_cpu()
                 logger.info("spaCy настроен для использования CPU.")
            except Exception as e_cpu:
                 logger.error(f"Не удалось даже явно указать CPU для spaCy: {e_cpu}")
    else:
        logger.info("Настройка использования CPU для spaCy.")
        try:
            spacy.require_cpu()
            logger.info("spaCy настроен для использования CPU.")
        except Exception as e:
            logger.error(f"Не удалось явно указать CPU для spaCy: {e}")

# --- Проверка моделей (остается синхронной) ---
def check_models() -> bool:
    """Проверяет наличие необходимых NLP моделей."""
    stanza_ok = False
    spacy_ru_ok = False
    spacy_en_ok = False

    logger.info("Проверка наличия NLP моделей...")
    logger.info(f"Настройка USE_GPU в config.py: {USE_GPU}")

    # Проверка Stanza (синхронно)
    try:
        logger.info(f"Проверка Stanza для языка '{LANGUAGE_CODE}'...")
        stanza.Pipeline(lang=LANGUAGE_CODE, processors='tokenize,ner', logging_level='WARN', use_gpu=USE_GPU, download_method=None)
        logger.info(f"Модель Stanza (с NER) для языка '{LANGUAGE_CODE}' найдена и инициализирована.")
        stanza_ok = True
    except FileNotFoundError:
         logger.error(f"Модель Stanza для языка '{LANGUAGE_CODE}' не найдена или не содержит NER.")
         logger.error(f"Пожалуйста, загрузите модель. Выполните в Python: import stanza; stanza.download('{LANGUAGE_CODE}')")
         stanza_ok = False
    except ImportError:
         logger.error(f"Ошибка импорта при инициализации Stanza. Возможно, проблема с CUDA при USE_GPU=True.")
         logger.error("Попробуйте установить USE_GPU = False в config.py или проверьте установку CUDA.")
         stanza_ok = False
    except Exception as e:
        logger.error(f"Не удалось проверить модель Stanza для языка '{LANGUAGE_CODE}'. Ошибка: {e}", exc_info=True)
        stanza_ok = False

    # Проверка spaCy ru (синхронно)
    try:
        logger.info(f"Проверка spaCy модели '{SPACY_MODEL_RU}'...")
        nlp = spacy.load(SPACY_MODEL_RU)
        logger.info(f"Модель spaCy '{SPACY_MODEL_RU}' успешно загружена.")
        spacy_ru_ok = True
        del nlp
    except OSError:
        logger.error(f"Не удалось загрузить модель spaCy '{SPACY_MODEL_RU}'.")
        logger.error(f"Пожалуйста, загрузите модель. Выполните: python -m spacy download {SPACY_MODEL_RU}")
        spacy_ru_ok = False
    except Exception as e:
         logger.error(f"Ошибка при проверке модели spaCy '{SPACY_MODEL_RU}': {e}", exc_info=True)
         spacy_ru_ok = False

    # Проверка spaCy en (синхронно)
    try:
        logger.info(f"Проверка spaCy модели '{SPACY_MODEL_EN}'...")
        nlp_en = spacy.load(SPACY_MODEL_EN)
        logger.info(f"Модель spaCy '{SPACY_MODEL_EN}' найдена.")
        spacy_en_ok = True
        del nlp_en
    except OSError:
        logger.warning(f"Не удалось загрузить модель spaCy '{SPACY_MODEL_EN}'.")
        logger.warning(f"Для некоторых функций Presidio может потребоваться. Выполните: python -m spacy download {SPACY_MODEL_EN}")
        spacy_en_ok = True
    except Exception as e:
         logger.warning(f"Ошибка при проверке модели spaCy '{SPACY_MODEL_EN}': {e}")
         spacy_en_ok = True

    if not (stanza_ok and spacy_ru_ok):
        logger.error("Отсутствуют или не удалось инициализировать критически важные NLP модели (Stanza ru или SpaCy ru).")
        return False

    logger.info("Проверка моделей завершена.")
    return True

# --- Новая основная асинхронная функция ---
async def main_async(): # <-- async def
    logger.progress("="*20 + " Запуск скрипта анонимизации " + "="*20)

    # Настраиваем устройство ДО проверки моделей (синхронно)
    setup_spacy_device()

    # 1. Проверка наличия моделей (синхронно)
    if not check_models():
        logger.critical("Не удалось загрузить или инициализировать необходимые NLP модели. Завершение работы.")
        # В асинхронной функции нельзя использовать sys.exit(1), лучше выбросить исключение
        raise RuntimeError("Model check failed")

    # 2. Загрузка конфигурации из файлов (асинхронно)
    logger.info("Загрузка конфигурации...")
    # Используем await для асинхронных функций
    entities_to_process = await load_entities_to_process(ENTITIES_FILENAME)
    exceptions_list = await load_exceptions(EXCEPTIONS_FILENAME)

    # 3. Проверка списка сущностей (синхронно)
    if not entities_to_process:
        logger.error("Список сущностей для обработки пуст. Анонимизация невозможна.")
        raise RuntimeError("Empty entity list")

    # 4. Запуск основного процесса анонимизации (асинхронно)
    logger.progress("Запуск основного процесса анонимизации...")
    try:
        # Используем await для асинхронной функции
        await anonymize_text_file(
            input_file=INPUT_FILENAME,
            output_file=OUTPUT_FILENAME,
            entities_to_process=entities_to_process,
            exceptions_list=exceptions_list,
            language=LANGUAGE_CODE,
            spacy_model=SPACY_MODEL_RU
        )
        logger.progress("Основной процесс анонимизации успешно завершен.")
    except Exception as e:
        # Логируем ошибку внутри асинхронной функции
        logger.critical(f"Критическая ошибка во время выполнения основного процесса анонимизации: {e}", exc_info=True)
        # Перевыбрасываем исключение, чтобы его поймал внешний обработчик
        raise e

    logger.progress("="*20 + " Скрипт анонимизации завершил работу " + "="*20)


if __name__ == "__main__":
    exit_code = 0 # Код завершения
    try:
        # Запускаем основную асинхронную функцию через asyncio.run()
        asyncio.run(main_async())

    except RuntimeError as e:
        # Ловим ошибки, которые мы сами выбросили (например, при проверке моделей)
        logger.critical(f"Завершение работы из-за ошибки: {e}")
        exit_code = 1
    except Exception as main_error:
         logger.critical(f"Произошла непредвиденная ошибка в main:", exc_info=True)
         exit_code = 1

    finally:
        print("\n-----------------------------------------------------")
        input("Обработка завершена. Для закрытия окна нажмите Enter...")
        print("-----------------------------------------------------")
        sys.exit(exit_code) # Завершаем скрипт с соответствующим кодом