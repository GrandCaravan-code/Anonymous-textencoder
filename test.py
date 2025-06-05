import stanza
try:
    stanza.Pipeline(lang='ru', processors='tokenize,ner')
    print("Модель Stanza 'ru' с NER найдена.")
except Exception as e:
    print(f"Ошибка при проверке Stanza: {e}")
    print("Пытаюсь скачать модель Stanza 'ru'...")
    stanza.download('ru')
    print("Попробуйте запустить скрипт снова.")