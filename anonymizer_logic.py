# anonymizer_logic.py
"""
Содержит основную логику анонимизации с использованием Presidio и Natasha.
"""
import logging
import stanza
import spacy
import time # Для замера времени Natasha
import asyncio # <-- Добавлено для to_thread
import aiofiles
import re # <-- Добавлено для проверки паттернов в _adjust_ner_scores

# Импорты Presidio
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry, RecognizerResult, AnalysisExplanation
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import (
    EmailRecognizer, PhoneRecognizer, CreditCardRecognizer, IbanRecognizer,
    IpRecognizer, UrlRecognizer, StanzaRecognizer
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# --- НОВОЕ: Импорты Natasha ---
try:
    from natasha import (
        Segmenter,
        MorphVocab,
        NewsEmbedding,
        NewsMorphTagger,
        NewsNERTagger,
        Doc
    )
    NATASHA_AVAILABLE = True
    # Инициализация компонентов Natasha (остается синхронной при импорте)
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    ner_tagger = NewsNERTagger(emb)
    logging.info("Компоненты Natasha успешно инициализированы.")
except ImportError:
    logging.warning("Библиотека Natasha не найдена. NER с помощью Natasha не будет использоваться.")
    logging.warning("Установите ее: pip install natasha")
    NATASHA_AVAILABLE = False
    segmenter, morph_vocab, emb, morph_tagger, ner_tagger = None, None, None, None, None
# -----------------------------

# Импорты из других наших модулей
from config import ( # ИЗМЕНЕНО: Импортируем новые константы
    ENTITY_PLACEHOLDERS, DEFAULT_SCORE_THRESHOLD,
    ANCHOR_SCORE_THRESHOLD, NER_FILTER_LOW_SCORE_THRESHOLD,
    NER_LOW_CONFIDENCE_SCORE_MULTIPLIER, NATASHA_DEFAULT_SCORE
)
from custom_recognizers import create_custom_recognizers
from text_utils import post_process_text

# Список слов/фраз для дополнительной фильтрации NER-результатов
NER_FALSE_POSITIVE_FILTER = {
    # Роли и стороны (добавлены падежи)
    "сторона", "стороны", "стороне", "стороной", "сторон", "сторонам", "сторонами",
    "заказчик", "заказчика", "заказчику", "заказчиком", "заказчике",
    "исполнитель", "исполнителя", "исполнителю", "исполнителем", "исполнителе",
    "подрядчик", "подрядчика", "подрядчику", "подрядчиком", "подрядчике",
    "пользователь", "пользователя", "пользователю", "пользователем", "пользователе",
    "покупатель", "покупателя", "покупателю", "покупателем", "покупателе",
    "продавец", "продавца", "продавцу", "продавцом", "продавце",
    "генеральный директор", "представитель", "представителя", "представителю", "представителем", "представителе",
    "акт", "акта", "акту", "актом", "акте",
    "выписка", "выписки", "выписке", "выпиской", "выписку",
    "кадастровый", "кадастрового", "кадастровому", "кадастровым", "кадастровом",
    "квартира", "квартиры", "квартире", "квартирой", "квартиру",
    "договор", "договора", "договору", "договором", "договоре",
    # Общие термины
    "устав", "устава", "доверенность", "доверенности", "филиал", "общество",
    "центр", "отделение", "банк", "адрес", "реквизиты", "заявление",
    "увольнение", "кодекс", "федерация", "основание",
    "реализация", "право", "срок", "дата", "день", "подпись", "услуга",
    "текст", "лицо", "общественные финансы", "информационные технологии",
    "грузополучатель", "наименование", "место нахождения", "постановка",
    "учет", "орган", "россия", "российская федерация", "сша",
    "приложение", "пункт", "статья", "объект", "имущество", "паспорт",
    "серия", "номер", "выдан", "отдел", "отделение", "район", "город",
    "зарегистрирован", "зарегистрирована", "собственность", "реестр",
    "недвижимость", "запись", "момент", "заключение", "состояние",
    "здоровье", "обстоятельство", "сделка", "брак", "капитал", "дети",
    "обязанность", "документ", "ответственность", "случай", "признание",
    "сумма", "оплата", "стоимость", "налог", "платеж", "переход",
    "осмотр", "выявление", "недостаток", "уменьшение", "цена", "информация",
    "законодательство", "риск", "гибель", "повреждение", "передача",
    "порядок", "расчет", "регистрация", "орган", "расход", "доля",
    "исполнение", "обязательство", "сила", "извещение", "характер",
    "влияние", "срок", "последствие", "переговоры", "способ", "разрешение",
    "спор", "разногласие", "условие", "положение", "предмет", "залог",
    "притязание", "изменение", "дополнение", "форма", "экземпляр",
    "нотариус", "проверка", "система", "правило", "результат",
    "адрес", "реквизит", "подпись"
}

# --- Имена распознавателей для логики приоритета ---
STANZA_RECOGNIZER_NAME = "StanzaRecognizer"
SPACY_RECOGNIZER_NAME = "SpacyRecognizer"
NATASHA_RECOGNIZER_NAME = "NatashaRecognizer"
PRESIDIO_NLP_ENGINE_NAME = "NLP Engine (Presidio)"
NER_RECOGNIZER_NAMES = {SPACY_RECOGNIZER_NAME, STANZA_RECOGNIZER_NAME, PRESIDIO_NLP_ENGINE_NAME, NATASHA_RECOGNIZER_NAME}

# --- Пороги и множители теперь импортируются из config.py ---
# ANCHOR_SCORE_THRESHOLD
# NER_FILTER_LOW_SCORE_THRESHOLD
# NER_LOW_CONFIDENCE_SCORE_MULTIPLIER
# NATASHA_DEFAULT_SCORE

# --- Паттерн для проверки начала текста ---
KNOWN_LOWERCASE_PREFIX_PATTERN = re.compile(r"^(г|ул|просп|пер|пл|ш|б-р|наб|д|кв|корп|стр|пом|обл|р-н|пос|днп|снт|тер)\.?\s", re.IGNORECASE)


# --- Функция для запуска Natasha NER ---
def run_natasha_ner(text: str) -> list[RecognizerResult]: # Убран score_threshold как аргумент
    """
    Выполняет NER с использованием Natasha и возвращает результаты в формате Presidio.
    Использует NATASHA_DEFAULT_SCORE из config.py.
    Эта функция является СИНХРОННОЙ и блокирующей.
    """
    if not NATASHA_AVAILABLE:
        return []

    logger = logging.getLogger()
    logger.info("Запуск NER с помощью Natasha (в отдельном потоке)...")
    start_time = time.time()
    natasha_results = []
    try:
        doc = Doc(text)
        doc.segment(segmenter)
        if morph_tagger:
             doc.tag_morph(morph_tagger)
        else:
             logger.warning("Morphological tagger (Natasha) не инициализирован, морфологический анализ пропущен.")

        if ner_tagger:
            doc.tag_ner(ner_tagger)
        else:
             logger.warning("NER tagger (Natasha) не инициализирован, NER анализ пропущен.")
             return []

        type_mapping = {
            "PER": "PERSON",
            "LOC": "LOCATION",
            "ORG": "ORG"
        }

        for span in doc.spans:
            entity_type = type_mapping.get(span.type)
            if entity_type:
                # ИЗМЕНЕНО: Используем score из config.py
                score = NATASHA_DEFAULT_SCORE

                explanation = {
                    "recognizer_name": NATASHA_RECOGNIZER_NAME,
                    "original_score": score,
                    "text": span.text,
                    "natasha_type": span.type
                }

                result = RecognizerResult(
                    entity_type=entity_type,
                    start=span.start,
                    end=span.stop,
                    score=score,
                    analysis_explanation=explanation
                )
                natasha_results.append(result)
                logger.debug(f"  Natasha нашла: {entity_type} [{span.start}:{span.stop}] '{span.text}' (Score: {score:.2f})")

    except Exception as e:
        logger.error(f"Ошибка во время выполнения Natasha NER: {e}", exc_info=True)

    end_time = time.time()
    logger.info(f"Natasha NER завершен за {end_time - start_time:.2f} сек. Найдено {len(natasha_results)} сущностей (PER, LOC, ORG).")
    return natasha_results
# ------------------------------------------

# --- Функция для безопасного получения имени распознавателя (без изменений) ---
def _get_recognizer_info(result: RecognizerResult) -> tuple[str, str]:
    """Извлекает имя распознавателя и имя паттерна из результата."""
    recognizer_name = "N/A"
    pattern_name = "N/A"
    explanation = result.analysis_explanation

    if explanation:
        if isinstance(explanation, dict) and "recognizer_name" in explanation:
            recognizer_name = explanation.get("recognizer_name", NATASHA_RECOGNIZER_NAME)
        elif isinstance(explanation, AnalysisExplanation):
            if hasattr(explanation, 'recognizer_name_from_analyzer') and explanation.recognizer_name_from_analyzer:
                recognizer_name = explanation.recognizer_name_from_analyzer
            elif hasattr(explanation, 'recognizer_name') and explanation.recognizer_name:
                recognizer_name = explanation.recognizer_name
            elif hasattr(explanation, 'recognizer') and explanation.recognizer:
                 try:
                     recognizer_name = type(explanation.recognizer).__name__
                 except Exception:
                     recognizer_name = "Recognizer (Unknown Class)"
            elif hasattr(explanation, 'score') and not hasattr(explanation, 'pattern'):
                explanation_str = str(explanation)
                if 'Spacy' in explanation_str:
                    recognizer_name = SPACY_RECOGNIZER_NAME
                elif 'Stanza' in explanation_str:
                    recognizer_name = STANZA_RECOGNIZER_NAME
                else:
                    recognizer_name = PRESIDIO_NLP_ENGINE_NAME
            else:
                if hasattr(explanation, 'pattern') and isinstance(getattr(explanation, 'pattern', None), Pattern):
                    recognizer_name = "PatternRecognizer (Unknown Name)"
                else:
                    recognizer_name = "Unknown Recognizer"

            if hasattr(explanation, 'pattern_name') and explanation.pattern_name:
                pattern_name = explanation.pattern_name
            elif hasattr(explanation, 'pattern') and isinstance(getattr(explanation, 'pattern', None), Pattern):
                 try:
                     pattern_obj = getattr(explanation, 'pattern', None)
                     if isinstance(pattern_obj, Pattern):
                          pattern_name = pattern_obj.name
                 except Exception:
                      pattern_name = "Pattern (Error)"
        else:
             recognizer_name = f"Unknown Explanation Type ({type(explanation).__name__})"

    if recognizer_name in ["Recognizer (Unknown Class)", "Unknown Recognizer", "N/A"]:
         if "stanza" in str(result).lower():
              recognizer_name = STANZA_RECOGNIZER_NAME

    if recognizer_name == "PatternRecognizer (Unknown Name)" and pattern_name != "N/A":
         recognizer_name = f"PatternRecognizer ({pattern_name})"

    if recognizer_name == PRESIDIO_NLP_ENGINE_NAME and "Spacy" in str(explanation):
        recognizer_name = SPACY_RECOGNIZER_NAME

    return recognizer_name, pattern_name
# ----------------------------------------------------------------------

# --- Функция для проверки, является ли результат от NER (без изменений) ---
def is_ner_result(result: RecognizerResult) -> bool:
    """Проверяет, был ли результат получен от NER-модели."""
    recognizer_name, _ = _get_recognizer_info(result)
    return recognizer_name in NER_RECOGNIZER_NAMES
# ----------------------------------------------------------------------

# --- Вспомогательная функция для логирования списка результатов (без изменений) ---
def log_results_list(results: list[RecognizerResult], stage_name: str, text: str, logger: logging.Logger):
    """Логирует детали списка результатов на определенном этапе."""
    logger.debug(f"--- {stage_name} (Найдено: {len(results)}) ---")
    if not results:
        logger.debug("--- Список результатов пуст ---")
        return

    sorted_results = sorted(results, key=lambda x: x.start)

    for i, res in enumerate(sorted_results):
        recognizer_name, pattern_name = _get_recognizer_info(res)
        try:
            identified_text = repr(text[res.start:res.end])
        except IndexError:
            identified_text = "[Ошибка извлечения текста]"

        logger.debug(
            f"  {i+1}. Тип: {res.entity_type}, Score: {res.score:.3f}, Границы: [{res.start}:{res.end}], "
            f"Распознаватель: {recognizer_name} (Паттерн: {pattern_name if pattern_name != 'N/A' else 'NLP/Other'})"
            f"\n     Текст: {identified_text}"
        )
    logger.debug(f"--- Конец списка: {stage_name} ---")
# ----------------------------------------------------------------------

# --- Вспомогательная функция для определения "якоря" (без изменений) ---
def _is_anchor(result: RecognizerResult) -> bool:
    """Определяет, является ли результат 'якорным' (высококачественным)."""
    recognizer_name, _ = _get_recognizer_info(result)
    if recognizer_name == STANZA_RECOGNIZER_NAME:
        return True
    # Используем ANCHOR_SCORE_THRESHOLD из config
    if result.score >= ANCHOR_SCORE_THRESHOLD:
        return True
    return False
# ----------------------------------------------------------------------

# --- Вспомогательная функция для проверки пересечения (без изменений) ---
def _check_overlap(res1: RecognizerResult, res2: RecognizerResult) -> bool:
    """Проверяет, пересекаются ли диапазоны двух результатов."""
    return max(res1.start, res2.start) < min(res1.end, res2.end)
# ----------------------------------------------------------------------

# --- Функция для понижения score подозрительных NER результатов (без изменений в логике, использует константы из config) ---
def _adjust_ner_scores(
    results: list[RecognizerResult],
    text: str,
    logger: logging.Logger
) -> list[RecognizerResult]:
    """
    Понижает score для NER-результатов (PERSON, LOCATION, ORG),
    которые начинаются со строчной буквы (кроме известных исключений).
    Использует NER_LOW_CONFIDENCE_SCORE_MULTIPLIER из config.
    """
    adjusted_count = 0
    adjusted_results = []
    ner_types_to_adjust = {"PERSON", "LOCATION", "ORG"}

    for result in results:
        recognizer_name, _ = _get_recognizer_info(result)
        if recognizer_name in {SPACY_RECOGNIZER_NAME, NATASHA_RECOGNIZER_NAME} and result.entity_type in ner_types_to_adjust:
            try:
                res_text = text[result.start:result.end]
                if res_text and res_text[0].islower() and not KNOWN_LOWERCASE_PREFIX_PATTERN.match(res_text):
                    original_score = result.score
                    # Используем множитель из config
                    result.score *= NER_LOW_CONFIDENCE_SCORE_MULTIPLIER
                    adjusted_count += 1
                    logger.debug(
                        f"  Понижен score для '{res_text}' ({result.entity_type} [{result.start}:{result.end}], rec={recognizer_name}) "
                        f"с {original_score:.3f} до {result.score:.3f} (начинается со строчной буквы)."
                    )
            except IndexError:
                logger.warning(f"Ошибка индекса при проверке текста для результата: {result}")
            except Exception as e:
                 logger.warning(f"Ошибка при корректировке score для результата {result}: {e}")

        adjusted_results.append(result)

    if adjusted_count > 0:
        logger.info(f"Корректировка score: Понижен score для {adjusted_count} NER-результатов (строчная буква).")
    else:
        logger.debug("Корректировка score: Не найдено NER-результатов для понижения score (строчная буква).")

    return adjusted_results
# ----------------------------------------------------------------------


# --- Функция для объединения и фильтрации (без изменений в логике, использует константы из config) ---
def merge_and_filter_results(
    presidio_results: list[RecognizerResult],
    natasha_results: list[RecognizerResult],
    text_for_debug: str
) -> list[RecognizerResult]:
    """
    Объединяет результаты от Presidio и Natasha, используя двухпроходный метод.
    Эта функция является СИНХРОННОЙ.
    """
    logger = logging.getLogger()
    combined_results = presidio_results + natasha_results
    if not combined_results:
        return []

    combined_results.sort(key=lambda r: (r.start, -r.end))
    log_results_list(combined_results, "Объединенные и отсортированные результаты (перед слиянием)", text_for_debug, logger)

    potential_anchors = [res for res in combined_results if _is_anchor(res)]
    log_results_list(potential_anchors, "Потенциальные якоря", text_for_debug, logger)

    anchor_results = []
    processed_indices_in_combined = set()

    if potential_anchors:
        anchor_results.append(potential_anchors[0])
        try:
            processed_indices_in_combined.add(combined_results.index(potential_anchors[0]))
        except ValueError: pass

        for current_anchor in potential_anchors[1:]:
            last_anchor = anchor_results[-1]
            current_rec_name, _ = _get_recognizer_info(current_anchor)
            last_rec_name, _ = _get_recognizer_info(last_anchor)

            if _check_overlap(current_anchor, last_anchor):
                is_current_stanza = current_rec_name == STANZA_RECOGNIZER_NAME
                is_last_stanza = last_rec_name == STANZA_RECOGNIZER_NAME

                replace_last = False
                if is_current_stanza and not is_last_stanza:
                    replace_last = True
                elif not is_current_stanza and is_last_stanza:
                    replace_last = False
                elif current_anchor.score > last_anchor.score:
                    replace_last = True
                elif current_anchor.score < last_anchor.score:
                    replace_last = False
                elif (current_anchor.end - current_anchor.start) > (last_anchor.end - last_anchor.start):
                    replace_last = True

                if replace_last:
                    logger.debug(f"  Слияние якорей: Замена '{text_for_debug[last_anchor.start:last_anchor.end]}' ({last_rec_name}, {last_anchor.score:.2f}) на '{text_for_debug[current_anchor.start:current_anchor.end]}' ({current_rec_name}, {current_anchor.score:.2f})")
                    try: processed_indices_in_combined.remove(combined_results.index(last_anchor))
                    except (ValueError, KeyError): pass
                    anchor_results[-1] = current_anchor
                    try: processed_indices_in_combined.add(combined_results.index(current_anchor))
                    except ValueError: pass
                else:
                    logger.debug(f"  Слияние якорей: Пропуск '{text_for_debug[current_anchor.start:current_anchor.end]}' ({current_rec_name}, {current_anchor.score:.2f}) из-за конфликта с '{text_for_debug[last_anchor.start:last_anchor.end]}' ({last_rec_name}, {last_anchor.score:.2f})")
            else:
                anchor_results.append(current_anchor)
                try: processed_indices_in_combined.add(combined_results.index(current_anchor))
                except ValueError: pass

    log_results_list(anchor_results, "Результаты после слияния якорей", text_for_debug, logger)

    final_results = list(anchor_results)
    remaining_results = [res for i, res in enumerate(combined_results) if i not in processed_indices_in_combined]
    log_results_list(remaining_results, "Оставшиеся результаты (не якоря)", text_for_debug, logger)

    for current_result in remaining_results:
        conflicts_with_anchor = False
        for anchor in anchor_results:
            if _check_overlap(current_result, anchor):
                current_rec_name, _ = _get_recognizer_info(current_result)
                anchor_rec_name, _ = _get_recognizer_info(anchor)
                logger.debug(f"  Обработка остальных: Пропуск '{text_for_debug[current_result.start:current_result.end]}' ({current_rec_name}, {current_result.score:.2f}) из-за конфликта с якорем '{text_for_debug[anchor.start:anchor.end]}' ({anchor_rec_name}, {anchor.score:.2f})")
                conflicts_with_anchor = True
                break

        if conflicts_with_anchor:
            continue

        should_add = True
        if final_results:
             last_added_non_anchor = None
             for res in reversed(final_results):
                 if not _is_anchor(res):
                     last_added_non_anchor = res
                     break

             if last_added_non_anchor and _check_overlap(current_result, last_added_non_anchor):
                 if current_result.start >= last_added_non_anchor.start and current_result.end <= last_added_non_anchor.end:
                     should_add = False
                     logger.debug(f"  Обработка остальных: Пропуск (вложен) '{text_for_debug[current_result.start:current_result.end]}' в '{text_for_debug[last_added_non_anchor.start:last_added_non_anchor.end]}'")
                 elif last_added_non_anchor.start >= current_result.start and last_added_non_anchor.end <= current_result.end:
                     logger.debug(f"  Обработка остальных: Замена (содержит) '{text_for_debug[last_added_non_anchor.start:last_added_non_anchor.end]}' на '{text_for_debug[current_result.start:current_result.end]}'")
                     try:
                         final_results.remove(last_added_non_anchor)
                     except ValueError:
                         logger.warning("Не удалось удалить last_added_non_anchor при замене.")
                 else:
                      current_rec_name, _ = _get_recognizer_info(current_result)
                      last_rec_name, _ = _get_recognizer_info(last_added_non_anchor)
                      if current_result.score >= last_added_non_anchor.score:
                           logger.debug(f"  Обработка остальных: Пересечение, замена '{text_for_debug[last_added_non_anchor.start:last_added_non_anchor.end]}' ({last_rec_name}, {last_added_non_anchor.score:.2f}) на '{text_for_debug[current_result.start:current_result.end]}' ({current_rec_name}, {current_result.score:.2f}) (выше score)")
                           try:
                               final_results.remove(last_added_non_anchor)
                           except ValueError:
                               logger.warning("Не удалось удалить last_added_non_anchor при замене по score.")
                      else:
                           should_add = False
                           logger.debug(f"  Обработка остальных: Пересечение, пропуск '{text_for_debug[current_result.start:current_result.end]}' ({current_rec_name}, {current_result.score:.2f}) из-за конфликта с '{text_for_debug[last_added_non_anchor.start:last_added_non_anchor.end]}' ({last_rec_name}, {last_added_non_anchor.score:.2f}) (ниже score)")

        if should_add:
            final_results.append(current_result)

    final_results.sort(key=lambda r: r.start)
    logger.info(f"Объединение и фильтрация (2-проходный метод): Исходно {len(combined_results)}, Якорей {len(anchor_results)}, Финально {len(final_results)}")
    return final_results
# ------------------------------------------------------------------------------------


# --- Функция для фильтрации результатов с приоритетом NER (без изменений) ---
def filter_by_ner_priority(
    results: list[RecognizerResult],
    text_for_debug: str
) -> list[RecognizerResult]:
    """
    Фильтрует список результатов, отдавая приоритет NER.
    Удаляет Regex-результаты, если они пересекаются с любым NER-результатом.
    Эта функция является СИНХРОННОЙ.
    """
    if not results:
        return []

    logger = logging.getLogger()
    ner_results = [res for res in results if is_ner_result(res)]
    regex_results = [res for res in results if not is_ner_result(res)]

    if not ner_results or not regex_results:
        return results

    final_results = list(ner_results)
    discarded_count = 0

    for regex_res in regex_results:
        keep_regex = True
        regex_name, regex_pattern = _get_recognizer_info(regex_res)
        regex_text = text_for_debug[regex_res.start:regex_res.end]

        for ner_res in ner_results:
            if max(regex_res.start, ner_res.start) < min(regex_res.end, ner_res.end):
                keep_regex = False
                ner_name, _ = _get_recognizer_info(ner_res)
                ner_text = text_for_debug[ner_res.start:ner_res.end]
                logger.debug(
                    f"  Пропуск Regex результата '{regex_text}' ({regex_res.entity_type} [{regex_res.start}:{regex_res.end}], "
                    f"распознаватель: {regex_name}, паттерн: {regex_pattern}) "
                    f"из-за пересечения с NER результатом '{ner_text}' ({ner_res.entity_type} [{ner_res.start}:{ner_res.end}], "
                    f"распознаватель: {ner_name})"
                )
                discarded_count += 1
                break

        if keep_regex:
            final_results.append(regex_res)

    final_results.sort(key=lambda r: r.start)

    logger.info(f"Фильтрация по приоритету NER: {discarded_count} Regex результатов пропущено.")
    return final_results
# ----------------------------------------------------------------------

# --- Функция get_anonymizer_operators (без изменений) ---
def get_anonymizer_operators(entities: list[str]) -> dict[str, OperatorConfig]:
    """Создает словарь операторов для AnonymizerEngine на основе плейсхолдеров."""
    logger = logging.getLogger()
    operators = {}
    if "DEFAULT" in ENTITY_PLACEHOLDERS and ("DEFAULT" not in entities or not entities):
        operators["DEFAULT"] = OperatorConfig("replace", {"new_value": ENTITY_PLACEHOLDERS["DEFAULT"]})
        logger.debug("Добавлен оператор DEFAULT для замены.")

    processed_entities = set()
    for entity in entities:
        if entity in processed_entities:
            continue

        if entity in ENTITY_PLACEHOLDERS:
            placeholder = ENTITY_PLACEHOLDERS[entity]
            operators[entity] = OperatorConfig("replace", {"new_value": placeholder})
            processed_entities.add(entity)
        elif entity == "DEFAULT" and entity in ENTITY_PLACEHOLDERS:
             operators["DEFAULT"] = OperatorConfig("replace", {"new_value": ENTITY_PLACEHOLDERS["DEFAULT"]})
             processed_entities.add(entity)
        else:
            known_types = set(ENTITY_PLACEHOLDERS.keys()) | {"PERSON", "LOCATION", "ORG", "DATE_TIME", "NRP",
                             "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS", "URL",
                             "RU_QUOTED_DATE", "RU_ADDRESS_PART", "RU_POSTAL_CODE", "RU_IDENTIFIER"}
            if entity not in known_types:
                 logger.warning(f"Неизвестный тип сущности '{entity}' обнаружен. Плейсхолдер не будет создан (кроме DEFAULT).")
            elif entity != "DEFAULT":
                 logger.warning(f"Сущность '{entity}' не найдена в словаре плейсхолдеров ENTITY_PLACEHOLDERS. Для нее не будет создан оператор замены (кроме DEFAULT, если он активен).")
            processed_entities.add(entity)

    logger.info(f"Сконфигурированы операторы анонимизации (замена на плейсхолдеры): {list(operators.keys())}")
    return operators


# --- Основная функция анонимизации (асинхронная) ---
async def anonymize_text_file(
    input_file: str,
    output_file: str,
    entities_to_process: list[str],
    exceptions_list: set[str],
    language: str,
    spacy_model: str
) -> None:
    """
    Основная функция анонимизации текста из файла (асинхронная).
    Оркестрирует загрузку моделей, настройку Presidio, анализ (Presidio + Natasha),
    понижение score подозрительных NER, объединение результатов (с приоритетом Stanza),
    фильтрацию и замену.
    Блокирующие NLP операции выполняются в отдельных потоках.
    """
    logger = logging.getLogger()

    current_entities_to_process = list(entities_to_process)

    # --- Проверка списка сущностей ---
    if not current_entities_to_process:
        logger.warning("Список сущностей для обработки пуст (entities_to_process).")
        natasha_entities = {"PERSON", "LOCATION", "ORG"}
        if NATASHA_AVAILABLE:
            missing_natasha = list(natasha_entities - set(current_entities_to_process))
            if missing_natasha:
                 logger.warning(f"Добавляем сущности {missing_natasha} в список для обработки, т.к. их ищет Natasha.")
                 current_entities_to_process.extend(missing_natasha)

        if not current_entities_to_process:
             logger.error("Список сущностей пуст и Natasha недоступна или не ищет нужные типы. Анонимизация невозможна.")
             return

    try:
        # --- 0. Создание кастомных распознавателей ---
        custom_recognizers_list = create_custom_recognizers()
        logger.info(f"Создано {len(custom_recognizers_list)} пользовательских распознавателей.")

        # --- 1. Создание основного NLP Engine (spaCy) ---
        logger.info(f"Создание основного NLP Engine (spaCy) для языка: {language} с моделью {spacy_model}")
        spacy_engine = SpacyNlpEngine(models=[{"lang_code": language, "model_name": spacy_model}])
        logger.info("Основной NLP Engine (spaCy) успешно создан.")

        # --- 2. Создание и Наполнение Кастомного Реестра Распознавателей Presidio ---
        logger.info("Создание и наполнение кастомного RecognizerRegistry Presidio...")
        registry = RecognizerRegistry(supported_languages=[language])
        logger.info(f"RecognizerRegistry Presidio успешно создан для языков: {[language]}")

        # Добавление StanzaRecognizer
        stanza_supported = {"PERSON", "LOCATION", "ORG", "NRP", "DATE_TIME"}
        stanza_entities_to_use = list(set(current_entities_to_process) & stanza_supported)
        if stanza_entities_to_use:
            logger.info(f"Добавление StanzaRecognizer для сущностей: {stanza_entities_to_use}")
            try:
                stanza_recognizer = StanzaRecognizer(
                    supported_language=language,
                    supported_entities=stanza_entities_to_use
                )
                registry.add_recognizer(stanza_recognizer)
                logger.info(f"{type(stanza_recognizer).__name__} успешно добавлен в реестр Presidio.")
            except Exception as e:
                logger.error(f"Не удалось инициализировать StanzaRecognizer: {e}", exc_info=True)
        else:
            logger.info("StanzaRecognizer не используется в Presidio.")

        # Добавление встроенных распознавателей Presidio (Regex)
        added_built_in = []
        if "EMAIL_ADDRESS" in current_entities_to_process:
            registry.add_recognizer(EmailRecognizer(supported_language=language))
            added_built_in.append("EmailRecognizer")
        if "PHONE_NUMBER" in current_entities_to_process:
            custom_phone_exists = any(
                hasattr(rec, 'supported_entities') and rec.supported_entities and rec.supported_entities[0] == "PHONE_NUMBER"
                for rec in custom_recognizers_list
            )
            if not custom_phone_exists:
                 registry.add_recognizer(PhoneRecognizer(supported_language=language, context=["телефон", "номер", "тел"], default_score=0.6))
                 added_built_in.append("PhoneRecognizer (встроенный)")
                 logger.info("Добавлен встроенный PhoneRecognizer.")
            else:
                 logger.debug("Встроенный PhoneRecognizer не добавляется, т.к. есть кастомный.")

        if "CREDIT_CARD" in current_entities_to_process:
            registry.add_recognizer(CreditCardRecognizer(supported_language=language))
            added_built_in.append("CreditCardRecognizer")
        if "IBAN_CODE" in current_entities_to_process:
            registry.add_recognizer(IbanRecognizer(supported_language=language))
            added_built_in.append("IbanRecognizer")
        if "IP_ADDRESS" in current_entities_to_process:
            registry.add_recognizer(IpRecognizer(supported_language=language))
            added_built_in.append("IpRecognizer")
        if "URL" in current_entities_to_process:
            registry.add_recognizer(UrlRecognizer(supported_language=language))
            added_built_in.append("UrlRecognizer")

        if added_built_in:
            logger.info(f"Добавлены встроенные распознаватели Presidio (Regex): {', '.join(added_built_in)}.")

        # Добавление пользовательских распознавателей (Regex)
        added_custom_count = 0
        for recognizer in custom_recognizers_list:
            if hasattr(recognizer, 'supported_language') and recognizer.supported_language != language:
                logger.warning(f"Кастомный распознаватель {getattr(recognizer, 'name', type(recognizer).__name__)} имеет несовпадающий язык ({recognizer.supported_language} вместо {language}) и будет пропущен.")
                continue

            if hasattr(recognizer, 'supported_entities') and recognizer.supported_entities:
                 entity_name = recognizer.supported_entities[0]
                 if entity_name in current_entities_to_process:
                     registry.add_recognizer(recognizer)
                     logger.info(f"Добавлен кастомный распознаватель Presidio (Regex): {getattr(recognizer, 'name', type(recognizer).__name__)} для сущности {entity_name}")
                     added_custom_count += 1
                 else:
                      logger.debug(f"Кастомный распознаватель Presidio {getattr(recognizer, 'name', type(recognizer).__name__)} для сущности {entity_name} пропущен (нет в списке entities).")
            else:
                 logger.warning(f"Кастомный распознаватель Presidio {getattr(recognizer, 'name', type(recognizer).__name__)} не имеет указанной supported_entities.")


        logger.info(f"Добавлено {added_custom_count} пользовательских распознавателей Presidio (Regex).")
        logger.info("Кастомный RecognizerRegistry Presidio успешно наполнен.")

        # --- 3. Настройка Analyzer Engine Presidio ---
        logger.info("Инициализация Analyzer Engine Presidio с SpacyNlpEngine и кастомным реестром...")
        analyzer = AnalyzerEngine(
            nlp_engine=spacy_engine,
            registry=registry,
            supported_languages=[language],
            default_score_threshold=DEFAULT_SCORE_THRESHOLD
        )
        logger.info(f"Analyzer Engine Presidio успешно инициализирован (Default Score Threshold: {DEFAULT_SCORE_THRESHOLD}).")

        # --- 4. Настройка Anonymizer Engine (Presidio) ---
        logger.info("Инициализация Anonymizer Engine Presidio...")
        anonymizer = AnonymizerEngine()
        logger.info("Anonymizer Engine Presidio успешно инициализирован.")

        # --- 5. Чтение входного файла ---
        logger.info(f"Чтение входного файла: {input_file}")
        try:
            async with aiofiles.open(input_file, mode='r', encoding='utf-8') as f_in:
                text_to_anonymize_local = await f_in.read()
            logger.info(f"Файл '{input_file}' успешно прочитан (длина: {len(text_to_anonymize_local)} символов).")
        except FileNotFoundError:
            logger.error(f"Входной файл '{input_file}' не найден.")
            return
        except Exception as e:
            logger.error(f"Ошибка при чтении файла '{input_file}': {e}")
            return

        # --- 6. Анализ текста ---
        # --- 6.1 Анализ с помощью Presidio (ВЫНОСИМ В ПОТОК) ---
        logger.info(f"Запуск анализа текста с помощью Presidio (в отдельном потоке) для поиска сущностей: {current_entities_to_process}...")
        presidio_analyzer_results = await asyncio.to_thread(
            analyzer.analyze,
            text=text_to_anonymize_local,
            entities=current_entities_to_process,
            language=language,
            return_decision_process=True
        )
        logger.info(f"Анализ Presidio завершен.")
        log_results_list(presidio_analyzer_results, "Результаты Presidio Analyzer (до корректировки score)", text_to_anonymize_local, logger)

        # --- 6.2 Анализ с помощью Natasha (ВЫНОСИМ В ПОТОК) ---
        natasha_analyzer_results = []
        natasha_entities_to_find = list(set(current_entities_to_process) & {"PERSON", "LOCATION", "ORG"})
        if NATASHA_AVAILABLE and natasha_entities_to_find:
            logger.info(f"Запуск анализа Natasha (в отдельном потоке) для сущностей: {natasha_entities_to_find}...")
            natasha_analyzer_results = await asyncio.to_thread(
                run_natasha_ner,
                text_to_anonymize_local
                # score_threshold теперь берется из config внутри run_natasha_ner
            )
            log_results_list(natasha_analyzer_results, "Результаты Natasha NER (до корректировки score)", text_to_anonymize_local, logger)
        elif not NATASHA_AVAILABLE:
             logger.info("Анализ Natasha пропущен (библиотека недоступна).")
        else:
             logger.info("Анализ Natasha пропущен (сущности PERSON, LOCATION, ORG не запрошены).")

        # --- 6.3 Корректировка score подозрительных NER результатов ---
        logger.info("Корректировка score для подозрительных NER результатов (spaCy, Natasha)...")
        presidio_adjusted_results = _adjust_ner_scores(presidio_analyzer_results, text_to_anonymize_local, logger)
        natasha_adjusted_results = _adjust_ner_scores(natasha_analyzer_results, text_to_anonymize_local, logger)
        log_results_list(presidio_adjusted_results, "Результаты Presidio Analyzer (ПОСЛЕ корректировки score)", text_to_anonymize_local, logger)
        log_results_list(natasha_adjusted_results, "Результаты Natasha NER (ПОСЛЕ корректировки score)", text_to_anonymize_local, logger)


        # --- 6.4 Объединение и фильтрация ---
        logger.info("Объединение и фильтрация результатов (двухпроходный метод)...")
        merged_results = merge_and_filter_results(
            presidio_adjusted_results,
            natasha_adjusted_results,
            text_to_anonymize_local
        )
        log_results_list(merged_results, "Результаты после merge_and_filter_results (2-проходный)", text_to_anonymize_local, logger)

        # --- 6.5 Фильтрация с приоритетом NER над Regex ---
        logger.info("Применение фильтра приоритета NER...")
        prioritized_results = filter_by_ner_priority(merged_results, text_to_anonymize_local)
        logger.info("Фильтрация по приоритету NER завершена.")
        log_results_list(prioritized_results, "Результаты после filter_by_ner_priority", text_to_anonymize_local, logger)

        # --- 7. Фильтрация результатов с учетом исключений ---
        analyzer_results_filtered_exceptions = []
        if exceptions_list:
            logger.info(f"Применение фильтра исключений ({len(exceptions_list)} шт.)...")
            filtered_count_exc = 0
            for result in prioritized_results:
                identified_text = text_to_anonymize_local[result.start:result.end]
                if identified_text.strip().lower() in exceptions_list:
                    recognizer_name, _ = _get_recognizer_info(result)
                    logger.debug(f"  Результат '{identified_text}' ({result.entity_type} [{result.start}:{result.end}], score={result.score:.3f}, rec={recognizer_name}) пропущен из-за наличия в exceptions.txt.")
                    filtered_count_exc += 1
                else:
                    analyzer_results_filtered_exceptions.append(result)
            logger.info(f"Фильтрация исключений: {filtered_count_exc} результатов пропущено.")
        else:
            logger.info("Список исключений пуст или не загружен. Фильтрация исключений не применяется.")
            analyzer_results_filtered_exceptions = prioritized_results
        log_results_list(analyzer_results_filtered_exceptions, "Результаты после фильтрации исключений", text_to_anonymize_local, logger)

        # --- 8. Фильтрация ложных срабатываний NER ---
        analyzer_results_final_filtered = []
        logger.info(f"Применение фильтра ложных срабатываний NER (по списку NER_FALSE_POSITIVE_FILTER)...")
        filtered_count_ner = 0
        ner_types_to_filter = {"PERSON", "LOCATION", "ORG"}
        for result in analyzer_results_filtered_exceptions:
            recognizer_name, _ = _get_recognizer_info(result)
            apply_ner_filter = False
            is_ner_res = is_ner_result(result)
            is_target_type = result.entity_type in ner_types_to_filter

            if is_ner_res and is_target_type:
                identified_text = text_to_anonymize_local[result.start:result.end]
                cleaned_text = identified_text.strip().lower()

                if not cleaned_text or cleaned_text.isnumeric() or all(c in '.,!?;:()[]{}<>"\'`~@#$%^&*-_=+|\n\t ' for c in cleaned_text):
                     logger.debug(f"  Результат NER '{identified_text}' ({result.entity_type} [{result.start}:{result.end}], rec={recognizer_name}) пропущен, т.к. содержит только пунктуацию/пробелы/цифры.")
                     apply_ner_filter = True
                elif cleaned_text in NER_FALSE_POSITIVE_FILTER:
                    logger.debug(f"  Результат NER '{identified_text}' ({result.entity_type} [{result.start}:{result.end}], rec={recognizer_name}) пропущен из-за точного совпадения с фильтром NER_FALSE_POSITIVE_FILTER.")
                    apply_ner_filter = True
                elif ' ' not in cleaned_text and cleaned_text in NER_FALSE_POSITIVE_FILTER:
                     logger.debug(f"  Результат NER (одно слово) '{identified_text}' ({result.entity_type} [{result.start}:{result.end}], rec={recognizer_name}) пропущен, т.к. слово есть в NER_FALSE_POSITIVE_FILTER.")
                     apply_ner_filter = True
                # Используем NER_FILTER_LOW_SCORE_THRESHOLD из config
                elif (cleaned_text and cleaned_text[0].islower() and not KNOWN_LOWERCASE_PREFIX_PATTERN.match(identified_text)) or result.score < NER_FILTER_LOW_SCORE_THRESHOLD:
                    words_in_result = set(cleaned_text.split())
                    common_words = words_in_result.intersection(NER_FALSE_POSITIVE_FILTER)
                    if common_words:
                        logger.debug(f"  Результат NER '{identified_text}' ({result.entity_type} [{result.start}:{result.end}], rec={recognizer_name}, score={result.score:.3f}) пропущен (низкий score или строчная буква), т.к. содержит слова из фильтра: {common_words}.")
                        apply_ner_filter = True

            if apply_ner_filter:
                filtered_count_ner += 1
                continue
            analyzer_results_final_filtered.append(result)

        logger.info(f"Фильтрация ложных срабатываний NER: {filtered_count_ner} результатов пропущено.")
        log_results_list(analyzer_results_final_filtered, "Финальные результаты для анонимизации", text_to_anonymize_local, logger)


        # --- 9. Анонимизация текста ---
        processed_text = text_to_anonymize_local
        if analyzer_results_final_filtered:
            logger.info(f"Запуск анонимизации текста (в отдельном потоке)... Найдено {len(analyzer_results_final_filtered)} сущностей для замены.")
            final_entities_in_results = list(set(res.entity_type for res in analyzer_results_final_filtered))
            all_possible_entities = list(set(final_entities_in_results) | set(current_entities_to_process))
            operators = get_anonymizer_operators(all_possible_entities)

            presidio_anon_logger = logging.getLogger("presidio-anonymizer")
            original_level = presidio_anon_logger.level
            presidio_anon_logger.setLevel(logging.DEBUG)
            logger.debug("Уровень логирования 'presidio-anonymizer' временно установлен на DEBUG для отслеживания конфликтов.")

            anonymized_result = await asyncio.to_thread(
                anonymizer.anonymize,
                text=text_to_anonymize_local,
                analyzer_results=analyzer_results_final_filtered,
                operators=operators
            )
            processed_text = anonymized_result.text

            presidio_anon_logger.setLevel(original_level)
            logger.debug(f"Уровень логирования 'presidio-anonymizer' возвращен на {logging.getLevelName(original_level)}.")

            logger.info("Анонимизация завершена.")
        else:
            logger.info("Сущности для замены (после всех фильтраций) не найдены. Анонимизация не выполняется.")

        # --- 10. Пост-обработка текста ---
        logger.info("Выполнение пост-обработки текста...")
        final_text = post_process_text(processed_text)
        logger.info("Пост-обработка завершена.")

        # --- 11. Запись результата в выходной файл ---
        logger.info(f"Запись результата в файл: {output_file}")
        try:
            async with aiofiles.open(output_file, mode='w', encoding='utf-8') as f_out:
                await f_out.write(final_text)
            logger.info(f"Результат успешно записан в '{output_file}'.")
        except Exception as e:
            logger.error(f"Ошибка при записи в файл '{output_file}': {e}")

    except ImportError as e:
         if 'natasha' in str(e).lower() and not NATASHA_AVAILABLE:
              logger.error("Ошибка импорта Natasha. Пожалуйста, установите библиотеку: pip install natasha")
         else:
              logger.error(f"Критическая ошибка импорта в anonymizer_logic: {e}")
         raise
    except Exception as e:
        logger.error(f"Произошла непредвиденная ошибка во время выполнения anonymize_text_file:", exc_info=True)
        raise e
