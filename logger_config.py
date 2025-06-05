# logger_config.py
"""
Модуль для настройки логирования приложения.

Настраивает логирование в файл и цветной вывод в консоль.
"""
import logging
import sys

# --- Константы для цветов ANSI ---
# (Могут не работать на старых терминалах Windows, но должны в Windows Terminal, Linux, macOS)
RESET = "\033[0m"
RED = "\033[91m"        # Ярко-красный для ошибок
GREEN = "\033[92m"      # Ярко-зеленый для прогресса
BLUE = "\033[94m"       # Ярко-синий для DEBUG
YELLOW = "\033[93m"     # Ярко-желтый для WARNING
BRIGHT_WHITE = "\033[97m" # Ярко-белый для INFO и остального

# --- Пользовательский уровень для сообщений о прогрессе ---
PROGRESS_LEVEL_NUM = 25 # Между INFO (20) и WARNING (30)
PROGRESS_LEVEL_NAME = "PROGRESS"

# --- Форматтеры ---
LOG_FORMAT_FILE = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
LOG_FORMAT_CONSOLE = '%(asctime)s - %(levelname)s - %(message)s' # Упрощенный для консоли

class ColoredConsoleFormatter(logging.Formatter):
    """
    Кастомный форматтер для добавления цветов к логам в консоли.
    """
    def __init__(self, fmt=LOG_FORMAT_CONSOLE, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.level_colors = {
            logging.DEBUG: BLUE,
            PROGRESS_LEVEL_NUM: GREEN, # Наш кастомный уровень
            logging.INFO: BRIGHT_WHITE,
            logging.WARNING: YELLOW,
            logging.ERROR: RED,
            logging.CRITICAL: RED, # Тоже красный
        }

    def format(self, record):
        # Получаем цвет для уровня записи
        log_color = self.level_colors.get(record.levelno, BRIGHT_WHITE) # По умолчанию ярко-белый

        # Форматируем сообщение стандартным образом
        message = super().format(record)

        # Добавляем цвет и сброс
        return f"{log_color}{message}{RESET}"

def setup_logging(level=logging.INFO, log_file="log.txt"):
    """
    Настраивает корневой логгер для вывода в файл и цветной консоли.
    """
    # Регистрируем наш кастомный уровень PROGRESS
    logging.addLevelName(PROGRESS_LEVEL_NUM, PROGRESS_LEVEL_NAME)

    # Получаем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(level) # Устанавливаем минимальный уровень для ВСЕХ обработчиков

    # --- Обработчик для файла ---
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w') # 'w' для перезаписи при каждом запуске
        file_handler.setLevel(level) # Уровень для файла
        file_formatter = logging.Formatter(LOG_FORMAT_FILE)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # Если не удалось создать файл лога, выводим ошибку в stderr
        print(f"CRITICAL: Не удалось настроить логирование в файл '{log_file}': {e}", file=sys.stderr)


    # --- Обработчик для консоли ---
    console_handler = logging.StreamHandler(sys.stdout) # Вывод в стандартный вывод
    console_handler.setLevel(level) # Уровень для консоли
    console_formatter = ColoredConsoleFormatter(LOG_FORMAT_CONSOLE)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Добавляем удобную функцию для логирования прогресса
    def log_progress(self, message, *args, **kws):
        if self.isEnabledFor(PROGRESS_LEVEL_NUM):
            self._log(PROGRESS_LEVEL_NUM, message, args, **kws)

    logging.Logger.progress = log_progress # Добавляем метод .progress() к логгеру

    logging.info(f"Логирование настроено. Уровень: {logging.getLevelName(level)}. Вывод в файл: '{log_file}' и в консоль.")

# Пример использования (если запустить этот файл напрямую)
if __name__ == '__main__':
    setup_logging(level=logging.DEBUG)
    logging.debug("Это сообщение для отладки (синее).")
    logging.info("Это обычное информационное сообщение (ярко-белое).")
    logging.progress("Это важное сообщение о прогрессе (зеленое).") # Используем новый метод
    logging.warning("Это предупреждение (желтое).")
    logging.error("Это сообщение об ошибке (красное).")
    logging.critical("Это критическая ошибка (красное).")