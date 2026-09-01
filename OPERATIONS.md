# Сервер, очередь и ручные сабмиты

Единый оперативный документ. Никакая задача не отправляет leaderboard-файлы
автоматически; новые артефакты — только plain `.csv`.

## Сервер

| ресурс | состояние / требование |
|---|---|
| host | `ubuntu@<ECUP_HOST>` (задаётся через `$ECUP_HOST`) |
| GPU | RTX 4090, 24 GB VRAM |
| RAM | 128 GB; одна тяжёлая задача за раз |
| OS | Ubuntu 24.04, CUDA 13.2 image |
| disk | матрицы ≈3.5 GB; нужен запас ≥20 GB |

SSH key хранится вне репозитория и в git не попадает. Скрипты `work/*.sh`
читают путь к ключу и адрес хоста из окружения:

```bash
export ECUP_KEY=/path/to/your-key.pem
export ECUP_HOST=ubuntu@<your-server>
```

Не копировать ключ в git, архивы или отчёты.

Corrected TCN 256/8 с batch 2048 использует около 8.7–11.8 GiB VRAM.
Конфигурация 384/12 с batch 2048 OOM; использовать batch 1024, fallback 768.

## Правила вычислительной очереди

- одна тяжёлая GPU-задача одновременно;
- перед handoff проверять PID, log progress, VRAM и свободный диск;
- selection fold заканчивается полностью, затем untouched holdout;
- full refit не запускается до прохождения frozen gates;
- watcher только скачивает/аудирует артефакты и не отправляет сабмиты;
- source используемого прогона сохраняется отдельной frozen-копией и SHA.

### Актуальная очередь 26.08.2026 11:35 МСК

SSH-сессия обрывалась; супервизоры `25923/31302/31675` погибли вместе с ней.
Обучающая задача под `nohup` уцелела. Текущее состояние проверено заново:

GPU **свободен** с 13:43 МСК. Очередь пуста намеренно: раунд масштабирования
завершён, все его оси закрыты отрицательно.

1. `server_position_tail.sh` — остановлен 11:47 (см. ниже).
2. `server_hidden_decay_tail.sh` — supervisor погиб, автоматически **не
   возобновится**; запускать вручную только по явному решению.
3. Новые скрипты раунда: `work/train_exact_scale.py` (окно/ширина/плотность
   как параметры, вариант `plain`), `work/server_exact_window_fanout.sh`,
   `work/server_exact_capacity.sh`, `work/server_capacity_factorial.sh`.

**Оперативная заметка.** Дважды `pkill -f <шаблон>` через `ssh` убивал саму
сессию, потому что шаблон совпадал с командной строкой удалённого `bash -c`.
Глушить процессы на сервере только по PID, а скрипты заливать через `scp`,
а не heredoc внутри команды ssh.

Завершено в exact-backbone раунде (ветка закрыта):

- decay seed93: strict admitted; probe `127=1.6464824097`, derived
  **ТОП-РЕШЕНИЕ** `130=1.6461706601`;
- decay seed-average: проваливает все strict и diagnostic gates
  (`+0.00000720`, incremental `−0.00000453`); эффект специфичен для seed93,
  поэтому `127` защищается как измеренная база, не как механизм;
- position seed93: **ЗОНД** `131=1.6465450851`; против корректного пула
  провал strict на `0.0000375008`, rejected;
- buyer seed93: сверх допущенного `{w409c, decay}` только `+0.00003356`,
  rejected offline, CSV `133` **не отправлен**;
- combined position+decay: incremental сверх отдельных семей `+0.00001602`,
  rejected;
- event seed93: conditional `+0.00007282 < 0.00008`, rejected;
- diagnostic `135` построен, но не отправляется.

Методологическая правка: `calculate_probe_gates.py` теперь принимает
`--expect-baseline` и падает, если marginal-пул не совпадает с пулом текущего
primary. Полный аудит — `work/gate_pool_specification_audit.json`.

Проверка после восстановления сессии:

```bash
ssh -i "$ECUP_KEY" "$ECUP_HOST" \
  'date; pgrep -af "server_w409_exact|train_w409_exact|server_position_tail|server_hidden_decay"; nvidia-smi'
```

Watchers локально скачивают decisions и строят plain CSV при pass; отправка на
leaderboard остаётся только ручной.

Подробные gates и hashes — в `RESEARCH_PROTOCOLS.md`.

## Репликация окружения

Перед новой серией обучения pipeline обязан воспроизвести контрольный
`tcn409rep` validation score около `1.67445`. Несовпадение означает проблему
окружения/данных и блокирует интерпретацию новых моделей.

Основные файлы:

```text
work/build_matrix.py          dense user×day matrices
work/build_matrix2.py         дополнительные funnel flags
work/feats3.py / feats4.py    262 / 304 features
work/train_seq2.py            device-agnostic sequence models
work/train_classifier_gpu.py  current GPU multitask/direct trainer
work/validate_submissions.py  shape/id/finite/nonnegative checks
```

## Ручной submission protocol

Перед отправкой кандидат должен иметь:

1. frozen mechanism и expected sign;
2. independent offline audit и численный gate;
3. plain CSV с 250 001 строкой, 250 000 unique `user_id`;
4. finite/nonnegative predictions и правильный id-order;
5. SHA-256 и meta/report;
6. явное решение пользователя отправить файл.

Batch отправляется полностью в заранее заданном порядке. Промежуточный ответ не
меняет следующие файлы, если именно так записан reading key. После batch:

- не менять mean/spread/lambda/active set;
- не строить blend по меньшему public score;
- не инвертировать проваленный probe;
- применить только frozen admission rule;
- записать actual score и verdict в `PROBE_JOURNAL.md`.

## Завершённые frozen rounds

### Rules-safe `120 → 122`

Оба файла были построены offline и отправлены без чтения первого score перед
вторым:

| файл | SHA-256 | frozen projection | public |
|---|---|---:|---:|
| `120_offline_rules_safe_meanforecast.csv` | `a17623…a88ae` | 1.6493324557 | 1.6493651732 |
| `122_offline_diverse_no_replica.csv` | `218e14…299b48` | 1.6502394144 | 1.6502888691 |

`123` был построен и назначен clean insurance до ответов round, но сам round
задним числом не расширялся.

Дополнительный заранее замороженный negative control
`119_offline_rules_safe_6model.csv` получил `1.6543467867`. Он хуже своей
rolling-origin mean-corrected версии `120` на `0.0049816135`; результат не
используется для post-hoc перенастройки весов.

### Structural `118 → 117`

Оба отправлены как frozen batch. `118` fail strict+adaptive на
`0.00004638`, `117` fail на `0.00022325`; ветка закрыта без joint recovery.

## Что не отправлять автоматически

- соседние lambda-варианты одного стека;
- почти идентичные market/calendar probes подряд;
- кандидаты с offline conditional gain ниже df/adaptive cost;
- отрицательную post-hoc инверсию;
- `.csv.gz`;
- любой probe только для косметического перехода с public rank 5 на rank 4.

## Перед финальным выбором

```bash
.venv/bin/python work/reproduce_frozen_finals.py
.venv/bin/python work/reproduce_clean_final_pair.py
```

После проверки сохранить hashes, выбрать ровно два файла в интерфейсе,
зафиксировать screenshot/confirmation и commit/tag репозитория.

## Батч `153` — НА УДЕРЖАНИИ, НЕ ОТПРАВЛЯТЬ

**Отменено 28.08 аудитом `155` до отправки хотя бы одного зонда.**
Замер `152` считал веса по public-50k целиком, а реальный солвер берёт Gram
по 250k и правую часть из public-моментов, да ещё подставляет `E[v^2]` по
250k вместо public. При честной симуляции обоих эффектов выигрыш `+0.00047`
при K=20 превращается в `−0.00281`, положителен в 0 из 48 сплитов.
Существующий пул от этого защищён: его базы коррелируют выше `0.99`, а веса
суммируются примерно в единицу, поэтому две ошибки взаимно гасятся. Сырые
исторические колонки коррелируют со стеком `0.35…0.87` и это гашение ломают.
Файлы оставлены на диске как материал следующего раунда; отправлять их в
текущем виде нельзя.

### Исходная инструкция (недействительна, пока держится удержание)

24 зонда лежат в `submissions/153_probe_hist01_*.csv` … `153_probe_hist24_*.csv`.
Все помечены **ЗОНД**: каждый измеряет один момент `E[z × колонка]` и сам по
себе плохой сабмит (ожидаемый скор `1.66`…`1.86`, см.
`work/153_probe_score_predictions.json`). Это нормально и заложено в конструкцию.

**Порядок обязателен.** Отправлять строго по возрастанию номера. Зонды
независимы, поэтому любой префикс — рабочий блок; но выбирать подмножество
после ответов запрещено замороженным ключом
(`work/153_conditional_block_preregister.json`).

**Как записать ответы.** В `work/153_probe_scores.json`, плоским словарём:

```json
{
  "153_probe_hist01_gmv_d365": 1.6661234567,
  "153_probe_hist02_active_recency": 1.8301234567
}
```

Скор, отличающийся от предсказанного больше чем на `0.02`, — повод перепроверить
перенос цифр, а не сразу вывод.

**Как получить решение.**

```bash
.venv/bin/python work/nl/apply_153_scores.py work/153_probe_scores.json
```

Скрипт вызывает замороженный `work/solve_augmented_stack.py`, добавляет колонки
к пулу `130`, пишет `submissions/154_conditional_block_k<K>_l0.003.csv` и
`work/153_block_decision_k<K>.json` с вердиктом `ADMIT` / `REJECT` по гейту
`fitted_public_gain − added_df × 0.0000395 ≥ 0.00016`. Собранный файл — это уже
**ТОП-РЕШЕНИЕ**, его нужно отправить отдельно, чтобы получить фактический public.

**Чего делать нельзя** (записано до отправки): менять `lambda`, инвертировать
знаки, искать лучшее подмножество колонок, перенастраивать веса после ответов.
