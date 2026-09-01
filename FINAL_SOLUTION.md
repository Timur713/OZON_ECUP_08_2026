# Финальные модели, воспроизводимость и production-план

Единая model card для competition-primary, clean finals и production story.
Стратегия выбора двух слотов и formal-risk находятся в `STRATEGY.md`.

## Что отслеживается в репозитории

Отслеживаются ровно **два выбранных финала**:

| слот | файл | SHA-256 |
|---|---|---|
| 1 | `submissions/200_shape_anchor_l003.csv` | `77b428ca6af9e74ffbdf22749cacfc87e76048d5dceef80146d557dabb21c598` |
| 2 | `submissions/147_private_safe_nonnegative_current_l001.csv` | `0c61bd73bfd0f699a25d893d3ba3cb762016021e446eb153fcbeef6d0bba51c2` |

Остальные submission-файлы в release не включаются. Исторические ветки ниже
сохраняются как методологический audit trail, но не являются выбранными
финалами.

```bash
.venv/bin/python work/reproduce_frozen_finals.py      # 200, 147
.venv/bin/python work/reproduce_clean_final_pair.py   # 120, 123
.venv/bin/python work/reproduce_rules_safe_finals.py  # 120, 122
```

Первый скрипт строго проверяет SHA-256 канонических CSV и численно пересобирает
оба финала во временной директории. Межплатформенный solver проверяется по
явным допускам, поскольку BLAS и bounded optimizer не гарантируют byte-exact
результат на разных архитектурах.

## Архитектура решения

Цель модели — `E[log1p(GMV_30d) | history]`, потому что RMSLE равен L2 в
`log1p`. Исследовательская библиотека сочетает:

- GBDT по 262–304 агрегатным RFM/funnel/seasonality признакам;
- TCN по 17 дневным каналам с окнами 45…409 дней;
- direct log-GMV regression;
- hurdle head `P(GMV>0) × E[log1p(GMV)|GMV>0]`;
- auxiliary 7/14/30/60-day heads;
- calendar, market, event-cadence и survival эксперименты.

Окна дают больше полезной диверсификации, чем замена TCN на GRU или смена
head. Competition ensemble — offline batch teacher, а не обязательный serving-
граф.

## Competition-primary: `200`

`200_shape_anchor_l003.csv` — ridge `lambda=0.003`, в котором основной
shape-кандидат объединён с двумя структурно отличающимися sequence-ветками,
двумя exact-window вариантами и frozen measured-блоком. Решение выбрано после
сценарного штрафа за effective df и четыре адаптивно допущенные базы.

| поле | значение |
|---|---:|
| public | `1.6457819828` |
| frozen expected public | `1.6457998906` |
| effective df | `23.6606` |
| adjusted private estimate | `1.6468765767` |
| SHA-256 | `77b428ca6af9e74ffbdf22749cacfc87e76048d5dceef80146d557dabb21c598` |

## Исторический competition-primary: `130`

`130_private_safe_exact_decay_l003.csv` расширяет frozen measured ridge новой
базой `127`: это точный сильный `w409c` backbone с четырьмя нормированными
экспоненциальными pooling окнами hidden-state (`7/30/90/180` дней). Изменение
проверено до public на 96 user-splits: independent gain `+0.00028222`,
conditional beyond `w409c +0.00012312`, положительный вес 96/96. Probe прошёл
заранее замороженный strict gate, после чего ridge λ=.003 был пересобран один
раз и заморожен до собственного ответа.

| поле | значение |
|---|---|
| public | `1.6461706601` |
| frozen expected public | `1.6461788499` |
| empirical-private estimate | `1.6469898137` |
| SHA-256 | `60e3e8ec507fb566bb95d0e2b59db63149d014ebb3cf7496752ecafcbf46d138` |

Public prediction error `−0.00000819`; weights `130` после ответа не меняются.
Веса используют разрешённый leaderboard-moment механизм; добавленная база
отдельно прошла independent private-risk gate.

## Второй финальный слот: `147`

`147_private_safe_nonnegative_current_l001.csv` — nonnegative ridge
`lambda=0.001` по **актуальному** пулу, включающему допущенную базу `127`.
Конструкция была зафиксирована на актуальном пуле до public-ответа; перебора
после ответа не было.

| поле | значение |
|---|---:|
| public | `1.6466762024` |
| предсказанный public | `1.6466888567` |
| ошибка прогноза | `−0.0000126543` |
| effective df | `12.963` |
| активных компонент | `13` |
| empirical-private estimate | `1.6471882461` |
| SHA-256 | `0c61bd73bfd0f699a25d893d3ba3cb762016021e446eb153fcbeef6d0bba51c2` |

Роль — хедж против заниженной цены переноса public→private и adaptive-cost
основного решения. С учётом frozen adaptive penalty он обгоняет `200` при
`r ≈ 0.0000686`, то есть примерно `1.74×` нашей эмпирической оценки. Пара
`200+147`: corr `0.9996455`, RMS log-distance `0.04339`.

Что этот слот **не** покрывает: он делит с `200` frozen measured-блок, поэтому
структурный отказ механизма восстановления моментов ударит по обоим. Этот риск
принят сознательно; полный сценарный расчёт — в
`work/270_final_pair_decision.json`.

## Leaderboard-free ветка: `120` (НЕ выбрана в финал)

`120` был вторым слотом до 26.08 и заменён на `147`. Сценарный расчёт показал,
что `120` не даёт ничего, пока истинная цена переноса не превысит `≈3.94×`
нашей оценки, тогда как низко-df хедж окупается уже с `1.69×` — и делает это
ровно в той полосе, где решается попадание в топ-5. Обоснование замены —
в `work/historical_pair_decision_2026-08-26.json`. Ветка сохраняется как
исторический leaderboard-free contingency и аргумент защиты, но в финал не
идёт.

`120_offline_rules_safe_meanforecast.csv` не читает public scores, recovered
moments или competition target mean.

Форма — nonnegative ridge по шести семействам, пять активны:

- GBDT262;
- seed-average TCN120;
- TCN365 growing-anchor;
- исходный TCN409;
- независимая TCN409 replication;
- TCN180 two-head получает нулевой вес.

`lambda=0.001` выбран на 96 frozen `fit 50k → score 200k` historical splits.
Independent historical RMSLE `1.66646800`, full `1.66638482`.

Глобальный `mean(log1p)` прогнозируется отдельно rolling-origin моделью по
historical 30/60/90-day lags, trend и Fourier calendar с обязательным
30-дневным label-availability gap. Лучший backtest RMSE `0.1224668`, forecast
`2.3205353309`. Только после freeze он был сопоставлен с public-derived
`2.3232887`; ошибка `−0.00275337` не использовалась для настройки.

| поле | значение |
|---|---|
| public | `1.6493651732` |
| frozen projection | `1.6493324557` |
| projection error | `+0.0000327175` |
| SHA-256 | `a17623edb7de20da05cb2de682c6ed78a9b891de112645ef3e99d07c8b6a88ae` |

## Clean insurance: `123`

`123_offline_capped_w035.csv` сохраняет все шесть семейств `120`, но
ограничивает каждый вес диапазоном `[0,0.35]`. `w409c` упирается в cap на
96/96 splits вместо веса `0.637` у primary.

| поле | значение |
|---|---|
| historical RMSLE | `1.66704986` (`+0.00058186` к `120`) |
| corr / RMS log distance к `120` | `0.9997470 / 0.03556` |
| public | `1.6493033021` |
| frozen projection | `1.6492634481` |
| SHA-256 | `4a698bb23242ad19fd0edac4cab5c318c63544ea8012b97895f6fc86f36ac599` |

`123` был заморожен и назначен insurance до ответов `120/122`. Маленькая
public-победа над `120` не меняет роли, заданные independent 96-split audit.

Единый label-free аудит всех условных финальных пар воспроизводится командой
`.venv/bin/python work/audit_final_pair_diversity.py`; authoritative output —
`work/final_pair_diversity_audit.json`. Для clean-пары `120+123` значения —
`0.9997470 / 0.03556`.

## Severe model-shift reserve: `122`

`122_offline_diverse_no_replica.csv` полностью исключает доминирующую
TCN409-replication. Independent historical RMSLE `1.66908184`, full
`1.66900715`, corr с `120=0.9987523`, RMS log distance `0.07869`.

Public `1.6502888691`, frozen projection `1.6502394144`, SHA
`218e141ec943e634d5547b16a85e0e2acbfb678b17108c440a5eca3f84299b48`.
Измеренная цена полного удаления реплики `≈0.002614` RMSLE, поэтому `122` —
reserve, а не preferred clean second slot.

Сильно season-specific `121` прошёл внутренний same-season user audit, но после
заранее замороженного reading key получил public `1.7621018713`. Допустимое
frozen conditional-направление оказалось неположительным, поэтому ветка
отвергнута без инверсии, blend или переобучения. Это измеренное свидетельство,
что user-level gift shape 2025 не переносится на 2026, несмотря на устойчивость
внутри одного сезона.

Воспроизведение clean-пары:

```bash
.venv/bin/python work/reproduce_clean_final_pair.py
```

Воспроизведение frozen `120/122` round:

```bash
.venv/bin/python work/reproduce_rules_safe_finals.py
```

Обе команды проверяют 250 000 уникальных пользователей, порядок, finite /
nonnegative predictions и канонические SHA-256.

## Repo-review и лицензии

Воспроизводимый export использует только библиотеки с лицензиями, допускающими
свободное коммерческое применение: NumPy/SciPy (BSD family), Polars и LightGBM
(MIT), PyArrow и PyTorch (Apache-2.0/BSD/MIT components). CatBoost
(Apache-2.0) встречается только в research-скриптах и не нужен clean final.
Proprietary model/API/runtime в training или inference final-решений нет.

Перед передачей репозитория нужно сохранить pinned `requirements.txt`,
не включать SSH key или competition data в commit и приложить commit/tag,
validator output и hashes выбранных двух файлов.

Audit 25.08 обнаружил, что локальные `ecup_export*.tgz`, включая ранее tracked
`ecup_export.tgz`, содержат `export/data/train.parquet`. Ключей в архивах нет.
`.gitignore` теперь блокирует `export/data/`, export-архивы, `.pem` и новые
submission-файлы, но перед передачей репозитория tracked archive/data history
нужно удалить из review-ветки отдельной контролируемой операцией; локальные
копии без явного согласования не удалять.

## Production readiness

### Три слоя

1. **Research library:** все окна, seeds, heads и отрицательные гипотезы.
2. **Competition inference:** frozen teacher для batch scoring 250k клиентов.
3. **Production target:** один student либо 3–5 семейств с historical-only OOF
   calibration, feature registry, drift/cost monitors и rollback.

### Измеренная compression Pareto

`work/analyze_compact_stack.py` строит фиксированную Pareto-кривую относительно
актуального `130` teacher; authoritative output —
`work/compact_stack_pareto.json`. Кривая измеряет цену compression serving-
графа, но не разрешает post-hoc выбирать subset: ranking наследует public-fit
teacher, а просмотр всей кривой добавляет subset/adaptive bias.

### Путь к промышленной модели

1. Выбрать 3–5 семейств по rolling-origin, holiday holdout и cost constraint.
2. Получить temporal OOF teacher predictions.
3. Обучить student с loss
   `L2(student,target) + α·L2(student,teacher)`.
4. Калибровать только на historical OOF labels.
5. Версионировать schema/data cutoff/model и добавить rollback.
6. Мониторить input drift, buyer rate, mean/std `log1p(pred)`, latency и delayed
   outcome quality.

Команды аудита:

```bash
.venv/bin/python work/analyze_compact_stack.py --max-models 28
.venv/bin/python work/compare_compact_stack_risk.py --max-models 12
```
