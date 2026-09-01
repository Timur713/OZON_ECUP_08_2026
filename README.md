# E-CUP 2026 — Search User Value

Решение трека 3: прогноз суммарного GMV клиента за 30 дней по 409 дням
обезличенной активности в Поиске и Каталоге. Метрика — RMSLE, размер test —
250 000 пользователей, public/private split — 20%/80%.

## Финальная пара

Авторитетное решение зафиксировано в `work/270_final_pair_decision.json`:

| роль | файл | public RMSLE | SHA-256 |
|---|---|---:|---|
| quality | `submissions/200_shape_anchor_l003.csv` | `1.6457819828` | `77b428ca6af9e74ffbdf22749cacfc87e76048d5dceef80146d557dabb21c598` |
| low-df hedge | `submissions/147_private_safe_nonnegative_current_l001.csv` | `1.6466762024` | `0c61bd73bfd0f699a25d893d3ba3cb762016021e446eb153fcbeef6d0bba51c2` |

В зачёт идёт лучший из двух файлов. `200` максимизирует ожидаемое качество,
`147` ограничивает риск переноса public→private за счёт nonnegative ridge и
меньшей эффективной сложности.

## Быстрая проверка

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python reproduce_weights.py
```

Команда:

- проверяет канонические SHA-256 обоих финальных CSV;
- проверяет 250 000 уникальных `user_id`, их порядок и finite/nonnegative
  predictions;
- заново решает обе задачи во временной директории;
- сравнивает пересчёт с каноническими файлами в `log1p`-пространстве.

Канонические CSV побайтово зафиксированы. Численный solver не объявляется
побайтово переносимым между CPU/BLAS: обычный ridge воспроизводится примерно до
`1e-12` RMS, bounded nonnegative solve — до нескольких `1e-6` RMS. Допуски
явно записаны в `work/reproduce_frozen_finals.py`.

## Что находится в репозитории

- два выбранных plain-CSV финала;
- минимальный набор frozen `.npy`-тензоров для их численного аудита;
- metadata, gates и исследовательский журнал;
- training/evaluation scripts и pinned зависимости.

Сырые competition parquet, виртуальные окружения, логи, промежуточные модели и
остальные submission-файлы не публикуются. Исторические solver-снимки, из
которых при release-redaction удалены отдельные компоненты, явно помечены
`weights_complete: false` и не используются для воспроизведения финальной пары.

## Метод в одном абзаце

RMSLE равен L2 по `z = log1p(y)`, поэтому модели оценивают
`E[z | history]`. Библиотека объединяет TCN с окнами 45…409 дней и GBDT по
RFM/funnel/seasonality-признакам. Финальный стек решается регуляризованной
линейной системой по измеренным агрегированным моментам. Новые компоненты
допускались только после frozen rolling-origin и `fit 50k → score 200k` gates;
public/private-риск учитывался штрафом за эффективные степени свободы.

## Навигация

| файл | назначение |
|---|---|
| `METHOD.md` | метод и итоговые метрики |
| `FINAL_SOLUTION.md` | model cards финальной пары и воспроизводимость |
| `STRATEGY.md` | правила допуска и выбор двух слотов |
| `FINDINGS.md` | устойчивые научные выводы |
| `RESEARCH_PROTOCOLS.md` | frozen protocols и gates |
| `PROBE_JOURNAL.md` | хронология разрешённых экспериментов |
| `PITCH.md` | структура защиты и Q&A |
| `OPERATIONS.md` | локальный запуск и ручной submission protocol |

Отправка на leaderboard всегда выполнялась вручную; код репозитория не
обращается к leaderboard API.
