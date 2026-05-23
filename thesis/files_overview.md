# Структура файлов в `thesis/`

Краткое описание того, как организованы файлы тезиса и за что отвечает каждый из них.

## Файлы

### `conference.tex`
Главный документ тезиса (шаблон IEEE conference, класс `IEEEtran`).
- Сюда переносится финальный текст работы (Introduction, Related Work, Methods, Experiments, Results и т.д.).
- Используется BibTeX-библиография из `stenosis.bib`.
- В шаблоне всё ещё остаются placeholder-секции из IEEE (`Ease of Use`, `Prepare Your Paper`, ...) и `\cite{b1}`–`\cite{b7}` — их нужно будет вычистить вручную, когда основной текст будет готов.

### `literature_review.tex`
Черновик литературного обзора (документ-класс `article`).
- Содержит написанные секции **Introduction** и **Related Work**.
- Ссылки оформлены через `\cite{key}` с ключами из `stenosis.bib`.
- Используется как «исходник», из которого текст переезжает в `conference.tex`.

### `stenosis.bib`
Единая BibTeX-библиография проекта.
- Содержит все библиографические записи по статьям, упомянутым в обзоре (CADICA, ARCADE, STQD-Det, PS-STT, Stenosis-DetNet, LT-YOLO, ODySSeI, VasoMIM и т.д.).
- На этот файл ссылается `\bibliography{stenosis}` как в `literature_review.tex`, так и в `conference.tex`.

### `stenosis.tex`
Личные заметки по статьям (документ-класс `article`).
- Для каждой работы из `stenosis.bib` — короткое описание метода, ключевых модулей, результатов и ограничений.
- Используется как справочный материал при написании Related Work и обсуждений (не предполагается включать в финальный тезис).

## Соглашения по ключам цитирования

Все цитирования в тексте идут на ключи из `stenosis.bib`, например:
- `popov2022reviewmodernapproachescoronary`, `cardiovascular` — обзоры;
- `MOON2021105819`, `Danilov2021`, `attentionstenosis`, `aipoweredrealtime`, `choudhary2026odysseiopensourceendtoendframework`, `ZHAO2021104667`, `menezes2023`, `molenar`, `huang2026vasomim` — single-frame методы;
- `fvcm`, `PANG2021101900`, `HAN2023106546`, `STQD-Det`, `li2025ltyolo` — video-based методы;
- `jimenezpartinen2024cadica`, `popov2024arcade` — датасеты.

При переносе текста из `literature_review.tex` в `conference.tex` ключи цитирования остаются неизменными — благодаря этому достаточно подключить `\bibliography{stenosis}` в `conference.tex`.
