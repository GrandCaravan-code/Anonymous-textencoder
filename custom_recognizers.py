# custom_recognizers.py
"""
Содержит функцию для создания списка пользовательских распознавателей Presidio
на основе регулярных выражений (PatternRecognizer).
ИЗМЕНЕНО: Добавлен распознаватель для кадастровых номеров.
"""
import logging
from presidio_analyzer import Pattern, PatternRecognizer
from config import LANGUAGE_CODE # Импортируем код языка

# Контекстные слова для различных типов сущностей
ID_CONTEXT = ["инн", "кпп", "огрн", "бик", "р/с", "к/с", "счет", "реквизиты", "банк", "номер"]
ADDRESS_CONTEXT = ["адрес", "проживающий", "зарегистрирован", "улица", "город", "область", "район", "нахождения", "корреспонденции", "место жительства"]
PHONE_CONTEXT = ["телефон", "номер", "тел", "контакт", "факс"]
# Контекст для PERSON (оставляем, он может помочь точным паттернам)
PERSON_CONTEXT = [
    "фио", "ф.и.о.", "сотрудник", "гражданин", "заявитель", "директор",
    "генеральный директор", "от", "кому", "сторона", "представитель",
    "исполнитель", "заказчик", "абонент", "пользователь", "владелец",
    "руководитель", "главный бухгалтер", "инженер", "менеджер",
    "управляющий", "администратор", "клиент", "пациент", "студент",
    "преподаватель", "автор", "получатель", "отправитель", "свидетель",
    "обвиняемый", "потерпевший", "истец", "ответчик", "арендатор",
    "арендодатель", "продавец", "покупатель", "заемщик", "кредитор",
    "доверенность", "действующий от имени", "в лице",
    # Добавим слова, часто встречающиеся перед/после ФИО
    "имени", "именем", "лице", "гражданина", "гражданке"
]
# Контекст для кадастрового номера
CADASTRAL_CONTEXT = ["кадастровый", "кадастровый номер", "номер участка", "кадастровым номером"]


def create_custom_recognizers() -> list[PatternRecognizer]:
    """Создает и возвращает список пользовательских распознавателей."""
    recognizers = []

    # 1. Дата в кавычках (RU_QUOTED_DATE)
    ru_quoted_date_pattern = Pattern(
        name="Russian Quoted Date Pattern",
        regex=r'«?\d{1,2}»?\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}г?\.?',
        score=0.8
    )
    ru_quoted_date_recognizer = PatternRecognizer(
        supported_entity="RU_QUOTED_DATE",
        name="Russian Quoted Date Recognizer",
        patterns=[ru_quoted_date_pattern],
        supported_language=LANGUAGE_CODE
    )
    recognizers.append(ru_quoted_date_recognizer)
    logging.debug(f"Создан пользовательский распознаватель: {ru_quoted_date_recognizer.name}")


    # 2. Части адреса (RU_ADDRESS_PART)
    ru_address_part_pattern = Pattern(
        name="Russian Address Part Pattern v2",
        regex=r'(?i)\b(?:ул|улица|просп|проспект|пер|переулок|пл|площадь|ш|шоссе|б[- ]р|бульвар|наб|набережная|г|город|д|дом|кв|квартира|корп|корпус|стр|строение|пом|помещ|помещение|обл|область|р[- ]н|район|край|респ|республика|пос|поселок|днп|снт|тер|территория)\.?[ \t]+[\w\d \t\-«»"\'./А-Яа-я()№,]{1,120}(?=\b|[,\n;]|\d{6}|$)',
        score=0.7
    )
    ru_address_part_recognizer = PatternRecognizer(
        supported_entity="RU_ADDRESS_PART",
        name="Russian Address Part Recognizer v2",
        patterns=[ru_address_part_pattern],
        supported_language=LANGUAGE_CODE,
        context=ADDRESS_CONTEXT
    )
    recognizers.append(ru_address_part_recognizer)
    logging.debug(f"Создан пользовательский распознаватель: {ru_address_part_recognizer.name}")

    # 3. Индекс (RU_POSTAL_CODE)
    ru_postal_code_pattern = Pattern(
        name="Russian Postal Code Pattern",
        regex=r'\b(\d{6})\b', # 6 цифр на границе слова
        score=0.7
    )
    ru_postal_code_recognizer = PatternRecognizer(
        supported_entity="RU_POSTAL_CODE",
        name="Russian Postal Code Recognizer",
        patterns=[ru_postal_code_pattern],
        supported_language=LANGUAGE_CODE,
        context=["индекс", "почтовый"] + ADDRESS_CONTEXT
    )
    recognizers.append(ru_postal_code_recognizer)
    logging.debug(f"Создан пользовательский распознаватель: {ru_postal_code_recognizer.name}")

    # 4. Организации (ORG) - Regex
    org_keywords = r'(?:ООО|ЗАО|ОАО|ПАО|ИП|АО|ГКУ|МУП|ГУП|ФГУП|Фонд|Компания|Фирма|Банк|УФК|Министерство|Отделение|Общество|Учреждение|Акционерное общество)'
    org_name_part = r'(?:["«“](?:[^"»”])+["»”]|[\w\d][\w\d \t\-\.\(\)/]*[\w\d])'
    ru_org_pattern = Pattern(
        name="Russian Organization Pattern v3",
        regex=rf'(?i)\b{org_keywords}[ \t]+{org_name_part}(?=\s|[,;\(\)\.]|$|\n)',
        score=0.85
    )
    ru_org_simple_pattern = Pattern(
        name="Russian Simple Organization Pattern",
        regex=r'\b(АО|ПАО|ООО|ЗАО)[ \t]+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*\b',
        score=0.75
    )
    ru_org_recognizer = PatternRecognizer(
        supported_entity="ORG",
        name="Russian Organization Recognizer v3",
        patterns=[ru_org_pattern, ru_org_simple_pattern],
        supported_language=LANGUAGE_CODE
    )
    recognizers.append(ru_org_recognizer)
    logging.debug(f"Создан пользовательский распознаватель: {ru_org_recognizer.name}")

    # 5. Идентификаторы (RU_IDENTIFIER)
    ru_inn_pattern = Pattern(name="Russian INN Pattern", regex=r'\b(\d{10}|\d{12})\b', score=0.9)
    ru_inn_recognizer = PatternRecognizer(supported_entity="RU_IDENTIFIER", name="Russian INN Recognizer", patterns=[ru_inn_pattern], supported_language=LANGUAGE_CODE, context=ID_CONTEXT)
    recognizers.append(ru_inn_recognizer)
    ru_kpp_pattern = Pattern(name="Russian KPP Pattern", regex=r'\b(\d{9})\b', score=0.9)
    ru_kpp_recognizer = PatternRecognizer(supported_entity="RU_IDENTIFIER", name="Russian KPP Recognizer", patterns=[ru_kpp_pattern], supported_language=LANGUAGE_CODE, context=ID_CONTEXT + ["кпп"])
    recognizers.append(ru_kpp_recognizer)
    ru_ogrn_pattern = Pattern(name="Russian OGRN Pattern", regex=r'\b(\d{13}|\d{15})\b', score=0.9)
    ru_ogrn_recognizer = PatternRecognizer(supported_entity="RU_IDENTIFIER", name="Russian OGRN Recognizer", patterns=[ru_ogrn_pattern], supported_language=LANGUAGE_CODE, context=ID_CONTEXT + ["огрн"])
    recognizers.append(ru_ogrn_recognizer)
    ru_bik_pattern = Pattern(name="Russian BIK Pattern", regex=r'\b(\d{9})\b', score=0.9)
    ru_bik_recognizer = PatternRecognizer(supported_entity="RU_IDENTIFIER", name="Russian BIK Recognizer", patterns=[ru_bik_pattern], supported_language=LANGUAGE_CODE, context=ID_CONTEXT + ["бик", "банк"])
    recognizers.append(ru_bik_recognizer)
    ru_account_pattern = Pattern(name="Russian Account Pattern", regex=r'\b(\d{20})\b', score=0.9)
    ru_account_recognizer = PatternRecognizer(supported_entity="RU_IDENTIFIER", name="Russian Account Recognizer", patterns=[ru_account_pattern], supported_language=LANGUAGE_CODE, context=ID_CONTEXT + ["р/с", "к/с", "счет", "расч/сч", "корр/сч"])
    recognizers.append(ru_account_recognizer)
    logging.debug(f"Созданы пользовательские распознаватели для RU_IDENTIFIER")


    # 6. Телефонные номера (PHONE_NUMBER)
    phone_pattern_1 = Pattern(name="Russian Phone Pattern (Formatted +7)", regex=r'\+7[ \t]?\(?\d{3}\)?[ \t]?\d{3}[- \t]?\d{2}[- \t]?\d{2}\b', score=0.95)
    phone_pattern_2 = Pattern(name="Russian Phone Pattern (8 XXX)", regex=r'\b8[ \t]?\(?\d{3}\)?[ \t]?\d{3}[- \t]?\d{2}[- \t]?\d{2}\b', score=0.85)
    phone_pattern_3 = Pattern(name="Russian Phone Pattern (10 digits - low confidence)", regex=r'\b\d{10}\b', score=0.4)
    phone_pattern_4 = Pattern(name="Russian Phone Pattern (+7 XXX XXX XX XX)", regex=r'\+7[ \t]?\d{3}[ \t]?\d{3}[ \t]?\d{2}[ \t]?\d{2}\b', score=0.9)
    ru_phone_recognizer = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        name="Custom Russian Phone Recognizer v2",
        patterns=[phone_pattern_1, phone_pattern_2, phone_pattern_4, phone_pattern_3],
        supported_language=LANGUAGE_CODE,
        context=PHONE_CONTEXT
    )
    recognizers.append(ru_phone_recognizer)
    logging.debug(f"Создан пользовательский распознаватель: {ru_phone_recognizer.name}")

    # 7. Кастомный распознаватель для PERSON (ФИО) - ТОЛЬКО ВЫСОКОТОЧНЫЕ ПАТТЕРНЫ
    surname_part = r'[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?'
    name_part = r'[А-ЯЁ][а-яё]+'
    initials_part = r'[А-ЯЁ]\.'

    # Паттерн 1: Фамилия И. О. (Высокая уверенность)
    person_pattern_surname_io = Pattern(
        name="Russian Name Pattern (Surname I.O.)",
        regex=rf'\b{surname_part}\s+{initials_part}\s?{initials_part}\b',
        score=0.95
    )

    # Паттерн 2: Фамилия Имя Отчество (с явными окончаниями -вич/-вна) (Высокая уверенность)
    person_pattern_fio_strict_endings = Pattern(
        name="Russian Name Pattern (Strict FIO endings)",
        regex=rf'\b{surname_part}\s+{name_part}\s+{name_part}(?:вич|вна)\b',
        score=0.95
    )

    # Паттерн 3: И. О. Фамилия (Высокая уверенность)
    person_pattern_io_surname = Pattern(
        name="Russian Name Pattern (I.O. Surname)",
        regex=rf'\b{initials_part}\s?{initials_part}\s?{surname_part}\b',
        score=0.9
    )

    # Создаем распознаватель PERSON только с высокоточными паттернами
    ru_person_recognizer = PatternRecognizer(
        supported_entity="PERSON",
        name="Custom Russian Person Recognizer v4 (High Confidence Only)", # Новое имя
        patterns=[
            person_pattern_surname_io,          # Ф И.О. (0.95)
            person_pattern_fio_strict_endings,  # Ф И О (-вич/-вна) (0.95)
            person_pattern_io_surname,          # И.О. Ф (0.9)
        ],
        supported_language=LANGUAGE_CODE,
        context=PERSON_CONTEXT # Контекст все еще может помочь этим паттернам
    )
    recognizers.append(ru_person_recognizer)
    logging.debug(f"Создан пользовательский распознаватель: {ru_person_recognizer.name} с {len(ru_person_recognizer.patterns)} паттернами.")

    # 8. Кадастровый номер (RU_CADASTRAL_NUMBER)
    ru_cadastral_pattern = Pattern(
        name="Russian Cadastral Number Pattern",
        # XX:XX:ЦИФРЫ:ЦИФРЫ на границе слова
        regex=r'\b\d{2}:\d{2}:\d+:\d+\b',
        score=0.95 # Высокая уверенность из-за специфичности формата
    )
    ru_cadastral_recognizer = PatternRecognizer(
        supported_entity="RU_CADASTRAL_NUMBER",
        name="Russian Cadastral Number Recognizer",
        patterns=[ru_cadastral_pattern],
        supported_language=LANGUAGE_CODE,
        context=CADASTRAL_CONTEXT # Добавляем контекстные слова
    )
    recognizers.append(ru_cadastral_recognizer)
    logging.debug(f"Создан пользовательский распознаватель: {ru_cadastral_recognizer.name}")

    logging.info(f"Создано {len(recognizers)} пользовательских распознавателей.") # Теперь их 12
    return recognizers