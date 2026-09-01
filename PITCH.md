# Питч и защита решения

Цель пяти минут — доказать качество, честность проверки, воспроизводимость и
практическую применимость, а не перечислить все модели.

## История на шесть слайдов

### 1. Задача и правильная статистическая цель

Прогноз 30-дневного GMV 250 000 клиентов по 409 дням активности. RMSLE — L2 в
`log1p`, поэтому моделируется `E[log1p(GMV)|history]`, а не средний GMV в
рублях.

### 2. Почему задача в основном классификационная

Около 82% измеренной достижимой дисперсии связано с событием «купит / не
купит». Поэтому совмещены direct log-GMV head и hurdle-разложение
`P(buy) × E[log1p(GMV)|buy]` с auxiliary горизонтами 7/14/30/60.

### 3. Представление истории и feature engineering

Плотная матрица `user × day`, TCN/Conv1d, окна 45…409, RFM/funnel/trend/
intermittency признаки, известный календарь будущего окна и leakage-safe
historical summaries. Пропущенный день означает отсутствие активности, а не
unknown value. Главное измеренное разнообразие создаёт длина receptive field.

### 4. Честная проверка

Rolling-origin без случайного перемешивания времени; user-holdout на точном
подарочном сезоне; repeated `fit 50k → score 200k` stress-tests; hypotheses и
gates фиксируются до public. Показать отрицательный кейс: classifier-модели
улучшали январский validation, но провалили frozen public gates и были
исключены без инверсии.

### 5. Финальное решение и risk control

Финальная competition-пара: `200` — shape-anchor ridge с максимальным
измеренным качеством; `147` — nonnegative low-df hedge. Public RMSLE:
`1.6457820` и `1.6466762`; corr `0.9996455`, RMS log-distance `0.04339`.
Leaderboard-free ветка `120/123` сохраняется как production contingency и не
маскируется под выбранные competition-файлы.

### 6. Воспроизводимость и production

Одна команда проверяет канонические SHA, 250 000 unique users, порядок,
finite/nonnegative values и численно пересобирает `200/147` во временной
директории. Competition ensemble — batch teacher; production target —
distilled student или 3–5 моделей с historical OOF calibration и drift/cost
monitoring.

## Таблица ключевых гипотез

| гипотеза | инструмент | итог |
|---|---|---|
| уровень таргета неизвестен | 2 scalar probes | `E[z]=2.3232887`, `E[z²]=10.7633307` |
| сезонный YoY перенос даёт уровень | сравнение с независимым moment | завышение 6.1% |
| обычная validation ранжирует модели | шесть моделей с public | Spearman `−0.086`, без выброса `−0.900` |
| смена архитектуры даёт diversity | GRU/two-head/CatBoost | corr `0.996–0.999`, вклад около нуля |
| длина окна даёт diversity | окна 45…409 | основная работающая ось |
| gift multiplier переносится | historical analogue | ухудшает RMSLE |
| stable residual curve велика | historical adjacent windows | крупный unconditional signal не найден; тест не является потолком |
| monotone survival улучшает `w409c` | untouched + 96 split | conditional `+0.0000357`, gate fail |
| exact event cadence полезна | matched ablation | эксперимент выполняется |

## Демонстрация репозитория

Показывать короткий воспроизводимый путь:

```bash
.venv/bin/python work/reproduce_frozen_finals.py
```

Затем открыть `FINAL_SOLUTION.md`, hashes и validator report. Не запускать
часовой GPU-run во время защиты.

## Вероятные вопросы жюри

- **Почему нет leakage?** Target windows разделены во времени; evaluated users
  исключаются из user-cross-fit; календарь будущего известен, остальные
  признаки заканчиваются на anchor.
- **Вы подгонялись к public?** Competition-ветка использует aggregate
  moment recovery. Организатор письменно разрешил калиброваться по public любым
  способом и подтвердил случайный client split; это снимает rule-risk, но не
  statistical overfit. Поэтому competition-final заморожен, а clean `120/123`
  построены только по historical labels и до public-ответов.
- **Почему RMSLE?** Это точно L2 в `log1p`; условное среднее в этой шкале —
  Bayes-optimal prediction.
- **Почему ансамбль?** Разные окна дают independent residual signal; ridge
  учитывает корреляцию и снижает variance.
- **Почему большой ансамбль можно питчить?** Это research teacher для batch
  scoring; production-план — compression audit и distillation.
- **Почему два final-файла?** Один максимизирует expected quality, второй
  страхует concentration/formal/domain-shift risk; соседние lambda не дают
  реальной диверсификации.
- **Что оказалось бесполезным?** Прямой seasonal multiplier, Tweedie/ZILN для
  другой цели, public subset-search и несколько classifier branches.
- **Можно ли воспроизвести?** Да: plain CSV и exact canonical hashes,
  минимальные frozen tensors, validator и численный cross-platform rebuild с
  явными допусками.

## Что подготовить до защиты

- чистую архитектурную схему;
- один слайд с 6–8 ключевыми абляциями, включая отрицательные;
- public screenshot с явной пометкой «20%, не private»;
- backup-слайд с moment formula и public→private df audit;
- письменный ответ организаторов по moment recovery и лимиту;
- подтверждение выбранных двух finals и commit/tag review repository;
- ответы на вопросы длиной 20–30 секунд и пройденный тайминг рассказа.
