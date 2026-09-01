# Исследовательская методология и frozen-протоколы

Этот файл объединяет правила честной проверки и все активные/закрытые
leaderboard-free ветки. Численные артефакты лежат в `work/*.json`; public-score
не участвует ни в одном описанном здесь gate.

## Общая процедура проверки

1. **Ключ чтения до замера.** До запуска записываются гипотеза, механизм,
   ожидаемый знак, controls и численный gate.
2. **Никакого assessed-window leakage.** Любая статистика оцениваемого окна
   исключается из feature generation, training labels и calibration.
3. **Dose-response.** Для потенциально протекающей оси проверяется несколько
   уровней усечения/разделения, а не один удобный split.
4. **Negative control** проверяет оцениватель; **positive control** подтверждает,
   что он способен увидеть эффект нужного масштаба.
5. Несущее число считается двумя независимыми путями, когда это возможно.
6. До сообщения эффект переводится в RMSLE/private-gain и сравнивается с ценой
   дополнительного df и adaptive selection.
7. Решение принимается по untouched temporal fold и фиксированным
   `fit 50k → score 200k` user-splits; completion задачи не означает promotion.
8. Прошедшая ветка разрешает full refit и подготовку plain CSV, но никогда не
   автоматическую отправку.

Иерархия доказательств: untouched out-of-time holdout > repeated independent
user-splits > matched ablation > обычная validation > fitted-public. Один
январский anchor используется для early stopping внутри прогона, но не для
сравнения архитектур: его ранжирование исторически инвертировалось на public.

## Аудит кривой остатков

Исторический тест проверял корреляцию остатков одного клиента в соседних
30-дневных окнах после лёгкого бустинга. На лагах 30–120 большой стабильный
безусловный user-level residual signal не найден. Это полезный отрицательный
результат, но не доказанный потолок качества: тест слеп к условным, сезонным,
коротким и межпользовательским механизмам, а первоначальная реализация не была
полностью pair-safe.

Model-dependent скобка `R²≈0.593` предполагает разложение
`z = μ + φ + ε` с независимым `ε`; её нельзя выдавать за информационный предел.
Корректная формулировка для защиты: «в остатках лёгкого бустинга не найден
крупный стабильный повторяемый user-level signal на лагах 30–120».

## Multi-anchor residual growth — завершённая базовая схема

Frozen implementations:

- `work/train_multi_anchor_growth.py` —
  `805b3666084c4f78202b93e10670db9e1e34043629e529bd6288c4eead4e201a`;
- `work/train_residual_growth.py` —
  `b1c5eb7055eee0ae712bb603daa4a8e0e722da8303713f8a70e5cf020b4eaaa8`.

30-day direct+hurdle LightGBM использует 304 historical features. Folds
270/306/342 служат selection, fold 378 untouched; label допустим только при
`anchor+30≤fold`. На каждом fold 50k пользователей калибруют, независимые 200k
оценивают. Selection objective — mean RMSLE + `0.20 × (worst−mean)`.

Residual correction для fold использует только более ранние residual rows.
Кандидаты: last-residual persistence и shallow LightGBM со shrinkage
`0.10/0.25/0.50/0.75/1.00`. Gate: selection не ухудшается, untouched 378
улучшается минимум на `0.00015`.

## Monotone survival — rejected

Гипотеза: вместо независимых и иногда противоречивых 7/14/30/60 heads учить
неотрицательные hazard increments для `(0,7]`, `(7,14]`, `(14,30]`, `(30,60]`.
TCN409, width 256, 8 blocks, seed 1300, три эпохи; fold342 выбирает epoch/mix,
fold378 untouched.

Frozen source `work/train_classifier_gpu_survival_frozen.py`:
`c73a9a58177690afc4e401f90af86ee14a5d16263cec86f3a6487e7a73ee8255`.

Требовались: ноль monotonicity violations; fold378 не хуже `w409c` более чем
на `0.001`; individual gain `≥0.00010`, joint beyond `w409c ≥0.00008`,
positive `≥90/96`, negative weight `≤5%`.

Факт: monotonicity violations `0`, individual gain
`+0.00007523±0.00000230` (94/96), joint total `+0.00025306`, но conditional
gain beyond `w409c` лишь `+0.00003569`. Ветка не прошла gates; full/CSV не
создавались.

## Event cadence — rejected

Гипотеза: pooled TCN не получает явно exact recency и latest inter-event gap.
Для каждого из 17 каналов добавлены два bounded historical-only значения.
TCN409, width 256, 8 blocks, seed 1310, три эпохи, calendar + multi-window
summaries, standard 7/14/30/60, magnitude и direct heads. Fold342 выбирает
epoch/mix; fold378 untouched.

Frozen files:

- trainer `work/train_classifier_gpu_event_frozen.py` —
  `832ded891349db7d521bbb6f954ac9d0bc84f0986a4e1b56072dd2fe0f5e0a67`;
- builder `work/build_event_summary_holdout.py` —
  `c74c43572f4cd712c7d63874cadad39bda01f9ff6f2547f73b3b73a113e6c7b2`;
- server `work/server_event_summary_growth.sh` —
  `e124f1efdb55b13a399edece1a61f212d3fbc1838016ae9d44d0751da8ad3af1`.

Promotion gates: fold378 не хуже `w409c` более чем на `0.0005`; individual
gain `≥0.00010`; conditional beyond `w409c ≥0.00008`; positive `≥90/96`;
negative weight `≤5%`; finite/reproducible full и ненулевая residual distance.

Selection scores `1.769871 / 1.755142 / 1.757984`; был зафиксирован epoch 2,
hurdle mix `0.70`. На untouched fold378 frozen prediction получил `1.687899`
против same-split `w409c=1.667924`; допустимая граница `1.668424`, gate miss
`0.019475`. Позднее завершённый заранее frozen matched-control получил
`1.676376`, то есть cadence ухудшил same-seed architecture ещё на `0.011523`.
96-split individual gain `+0.00007077` ниже `0.00010`, а weight
отрицателен 96/96. Joint total с `w409c=+0.00026683`, conditional лишь
`+0.00004945 < 0.00008`, event-weight снова отрицателен 96/96.

Ветка отвергнута без full/CSV и без post-hoc inversion. Authoritative verdict:
`work/event409_growth_verdict.json`. Direct-only follow-up уже стартовал, чтобы
отделить возможную пользу features от interference auxiliary losses.

## Exact matched-control cadence — completed, standalone rejected

Контроль зафиксирован во время первой эпохи event-run, до первого его score.
Он идентичен по seed, календарю, summaries, anchors, split и schedule; отсутствуют
только exact event-cadence features.

Frozen files:

- builder `work/build_event_control_holdout.py` —
  `b4d799d98f96f88c1d94a08c1d99c2b307fd6754def0419ed9a0302249ac981a`;
- server `work/server_event_control.sh` —
  `bb74f2ff494c44aac54a5d4fa94e252205eb5082877efd574c4e9185e1caf778`.

Механизм cadence считается подтверждённым, только если event-model лучше
matched-control на untouched fold378 минимум на `0.00015` и добавляет минимум
`0.00005` conditional private-gain в 96-split audit. Эти gates дополняют, а не
заменяют общие promotion gates.

Факт seed1310: selection заморозил epoch1/mix0.60; untouched score
`1.6763764`, хуже same-split `w409c` на `0.0084520`. Single gain лишь
`+0.00000532`, вес отрицателен 82/96; в joint с `w409c` control-вес
отрицателен 65/96. Как самостоятельная семья control rejected, но остаётся
валидным причинным baseline для marked-event и relative-position.

## Direct-only event cadence — queued follow-up

Эта ветка зафиксирована после того, как event epoch1 offline выбрал direct
head, поэтому она считается offline-adaptive и имеет более строгие gates.
TCN409 обучается только direct log-GMV loss: class/magnitude weights `0`, direct
weight `1`; seed 1320, три эпохи, fold342 selection, fold378 untouched.

Frozen files:

- frozen trainer `work/train_classifier_gpu_direct_event_frozen.py` —
  `d39d3f475923dc43fd42d3292e76bb37ccff347b22817796d781b4d08db7b648`;
- builder `work/build_direct_event_holdout.py` —
  `f339a528085832e86e9f0bba5af1e993a2a90ecfa97ba9377f412f8e3d16ced8`;
- server `work/server_direct_event_growth.sh` —
  `6557d6937f631155737da5a4df365ae7fe01f821a7436664a3d51eff2dcd883d`.

Gates: fold378 no worse than `w409c+0.0005`; individual gain `≥0.00012`;
conditional beyond `w409c ≥0.00010`; positive `≥90/96`; negative weight `≤5%`.

Selection `1.761209 / 1.759540 / 1.782375` заморозил epoch2. На untouched
fold378 он получил `1.680793` против `w409c=1.667924` и максимума `1.668424`:
gate miss `0.012369`. Individual private gain лишь `+0.00006836 <0.00012`, вес
отрицателен 95/96. Joint gain с `w409c=+0.00023412`, но conditional beyond
`w409c` только `+0.00001675 <0.00010`, direct-weight отрицателен 93/96.
Ветка rejected без full/CSV/inversion; authoritative verdict:
`work/directevent409_growth_verdict.json`.

## Recent-event regularity profile — seed1310 rejected, replication frozen

Frozen 25.08.2026 after event fold342 selection, but before event fold378,
matched-control, direct-only или новых public-ответов. Поэтому ветка считается
offline-adaptive и имеет более строгие gates.

К исходным exact recency/latest-gap добавлены четыре bounded historical-only
признака на канал: mean последних четырёх event gaps, их coefficient of
variation, current overdue phase относительно собственного mean gap и отношение
latest gap к recent mean. Это проверяет персональный ритм/регулярность, а не
расширяет сеть или перебирает loss.

Design точно совпадает с event-cadence: TCN409, width256, 8 blocks, seed1310,
summary+calendar, три эпохи, fold342 selection и fold378 untouched. Отличается
только six-value event profile вместо two-value cadence.

Frozen files:

- frozen trainer `work/train_classifier_gpu_regularity_frozen.py` —
  `6a12a1062bea25350924fd96d02e65056b45d85078568648c68d1eaaf328d9b0`;
- builder `work/build_event_profile_holdout.py` —
  `c2aabf62d08e1aded39af82e79770cdd9cedfe310206f0c0089a6bf35a783ccd`;
- server `work/server_event_profile_growth.sh` —
  `e5457665cff7fc41a649674a082debb248a130fcd9a3e9d9d4518fa340575831`;
- watcher `work/wait_fetch_audit_event_profile.sh` —
  `1ab401597b0a8bb8650a72c286d1f94cc89597bc90c2b6238bf9daea9d19394d`.

Все условия обязательны: fold378 не хуже `w409c+0.0005`; individual gain
`≥0.00012`; conditional beyond `w409c ≥0.00010`; positive `≥90/96`; negative
weight `≤5%`. Дополнительно profile должен улучшить original cadence на
fold378 минимум на `0.00010` и дать conditional gain beyond cadence минимум
`0.00005`. Passing разрешает full/plain CSV, не автоматический submission.

Seed1310 fact: fold342 выбрал epoch1/mix0.4; untouched `1.6772641287` против
same-split `w409c=1.6679244053` (`−0.0093397234`). Profile улучшил уже
проваленный cadence на `0.0106352`, но single audit дал отрицательный вес
96/96; в joint с `w409c` regularity-вес отрицателен 95/96, conditional gain
лишь около `0.0000281`. Против causal matched-control `1.6763764` regularity
также хуже на `0.0008877`. Все decisive gates провалены, поэтому семья закрыта
без full/CSV. Seed2718 отменён до старта: повтор уже однозначно отрицательной
ветки не добавлял бы полезной информации и только занимал GPU.

## Marked-event value profile — frozen before regularity result

Frozen 25.08.2026 после старта regularity, но до его epoch2/epoch3, untouched
результата и matched-control. Механизм сохраняет exact recency/latest gap и
заменяет четыре rhythm-статистики на bounded значения последних пяти ненулевых
событий: last value, recent mean, coefficient of variation и last/mean. Это
проверяет «метку» последнего purchase/funnel event, которую temporal pooling
может потерять, а не меняет width/loss/seed.

Design совпадает с regularity и control: TCN409, 17 каналов, width256/8 blocks,
summary+calendar, seed1310, fold342 selection → fold378 untouched, три эпохи.
Wrapper сохраняет прежнюю размерность six-value event profile и зависит от
неизменяемого regularity trainer.

Frozen hashes:

- wrapper `work/train_classifier_gpu_marked_frozen.py` —
  `703518202c0beb90da6a2e4ae0f6be534a33ab4326e3201d7aa03617516e9d4f`;
- dependency `work/train_classifier_gpu_regularity_frozen.py` —
  `6a12a1062bea25350924fd96d02e65056b45d85078568648c68d1eaaf328d9b0`;
- handoff `work/server_marked_event_growth.sh` —
  `a13620a67cd9a8b65f34cd76f361b87d9b86e5d9b680dd59d026c654e07c2eb1`;
- watcher `work/wait_fetch_audit_marked_event.sh` —
  `0fbe30d69527f4f68772d9184ef8534bcd10ab57c8d1ca7e9f4671725a57cbae`;
- reused builder `work/build_event_profile_holdout.py` —
  `c2aabf62d08e1aded39af82e79770cdd9cedfe310206f0c0089a6bf35a783ccd`.

Smoke-test проверил shape/finite и полный one-step train/predict. Все условия
обязательны: fold378 не хуже `w409c+0.0005`; individual gain `≥0.00012`;
conditional beyond `w409c ≥0.00010`; positive `≥90/96`; negative weight
`≤5%`. Дополнительно marked должен улучшить original cadence на fold378
минимум `0.00010` и дать conditional beyond cadence `≥0.00005`. Сравнение с
regularity диагностическое, потому что обе ветки заморожены до результатов;
никакого winner-picking без independent gates. Passing разрешает full/plain
CSV, не automatic submission.

## Pair-safe residual diagnostic — completed, rejected

Implementation `work/exp_residcurve_pairsafe.py`:
`1781013abb1a89396536f352683c916868d7b098032225170a55a49fb03f7a3a`.
Server handoff `work/server_pairsafe_residual.sh`:
`92e10520480be717f3cb773816019553e8dc74fc387d13fb285a06787ddd2b2d`;
fetch watcher `work/wait_fetch_pairsafe_residual.sh`:
`1dc584969e8043d30d572b47f48d500e901efa8ecabdb5f9b9399297c9eb2b96`.
25.08 после подтверждённого direct-only holdout fail wait-only handoff заменён
на CPU/GPU-parallel handoff `work/server_pairsafe_residual_parallel.sh`, hash
`9a3f7f59c19a000025a7f6f57147153751f882aea63d5c1bac92f5f66c872c78`.
Model code, data, seeds, 24 threads и gates не менялись.

Восемь adjacent non-overlapping 30-day pairs используют anchors
`138,168,…,378`. Для пары оба окна исключаются из training labels; двухфолдовый
user cross-fit исключает evaluated users из fitting; calibration берётся на
другой половине пользователей. Negative control переставляет второй residual,
positive control инжектирует `corr≈0.01`; uncertainty — two-way resampling по
200 user-blocks и восьми парам.

Run валиден только при `|negative mean|≤0.003` и positive mean в
`[0.006,0.014]`. Stable-user branch поддерживается лишь при mean correlation
`≥0.003`, положительном знаке минимум в 6/8 парах и bootstrap p05 выше нуля.
Passing разрешает новый rank/regularity experiment, но не сабмит.

Фактический результат: controls валидны (`negative=0.001168`,
`positive=0.011297`), но mean adjacent correlation `−0.000108`, положительный
знак лишь в 4/8 парах, two-way bootstrap p05 `−0.007219`. Все три signal-gate
провалены; `supports_stable_user_residual_branch=false`. Ветка закрыта без
инверсии или post-hoc выбора поздних пар. Authoritative report:
`work/residcurve_pairsafe_report.json`.

## Lagged-residual clean follow-up — preregistered before diagnostic

Frozen preregistration: `work/lagged_residual_followup_preregister.json`;
conditional builder: `work/build_lagged_residual_clean_candidate.py`. Оба
созданы 25.08.2026 до pair-safe результата, direct-only untouched результата и
любого нового public-ответа.

Frozen hashes: preregistration
`f86db88070ff3b01e6ecc97b6019620f5dfaca919d8dd6935d9720996bc44ba8`;
builder `b0e9b1e6af5203aa83d74ae184fd0c9ba1b83be608b9e30c610a33bafdb43e61`;
server handoff
`b9e72e1a956cb96ba38f2ddd3ee9a6b7e7254210dd6f21404b8ee7d37a51da3b`;
fetch watcher
`671540d701ab1fee9a32fed79e633798398a1802c9755cb38542333b621ed064`.

Механизм: если cross-fitted residual клиента устойчив между соседними
непересекающимися 30-дневными окнами, residual на anchor348 добавляется как
седьмая база к historical six-family stack для untouched target379:408. Для
competition final соответствующая база — residual anchor378, полностью
наблюдаемый к cutoff408. Коэффициент constrained nonnegative; lambda `0.001`,
семейства и rolling-origin mean forecast точно наследуются от clean `120`.

Trigger использует controls исходного diagnostic и только первые семь temporal
pairs: mean correlation `≥0.003`, положительный знак минимум 6/7. Последняя
пара `(348,378)` не участвует в trigger и служит temporal holdout.

Promotion gates без public: mean independent 50k→200k gain `≥0.00012`, gain
положителен минимум 90/96 splits, residual weight положителен минимум 90/96,
full validation gain `≥0.00010`, RMS log-distance от `120 ≥0.01`, затем
validator finite/nonnegative/unique/plain CSV. При любом fail builder пишет
только JSON verdict и не создаёт CSV. Inversion, lambda/subset search и
ослабление gates запрещены.

## Очередь и правило остановки

### Two-seed conditional full promotion — frozen before profile results

После завершения seed `1310/2718` для exact control и marked-event запускается
единый leaderboard-blind decision. Каждая семья обязана на обоих
seed одновременно иметь: single independent gain `≥0.00012`, положительный
gain минимум в `90%` split, отрицательный вес максимум в `10%`, conditional
gain сверх `w409c ≥0.00008` и тот же sign-gate в joint audit. Exact control
может проиграть `w409c` на untouched не более `0.00050`; feature-family должна
победить свой same-seed exact control минимум на `0.00020`.

Проходит не «лучшая из двух», а каждая семья, независимо выполнившая все gates
на обоих seed. Для прошедшей семьи два full-refit используют только заранее
выбранные fold342 epoch/mix; их стандартизованные vectors усредняются 50/50.
Только затем локальная сборка строит plain 30%-probe от frozen baseline; большой
базовый CSV не является серверной зависимостью. При fail нет full-refit, CSV,
инверсии, ослабления порога или выбора по public.

При суточном лимите 30 заранее добавлен отдельный diagnostic-tier, который не
даёт admission в final: на обоих seed нужны single gain `≥0.00004`, conditional
сверх `w409c ≥0.00002`, positive/sign `≥75%`; profile может уступить control не
более `0.00050`, control — `w409c` не более `0.00150`. Такой structural
near-pass получает ровно один frozen two-seed probe. Его public-ответ служит
только cross-moment measurement и ещё обязан покрыть empirical df + adaptive
public→private cost. Соседние lambda/mix по-прежнему запрещены.

Implementation: `work/evaluate_profile_promotion.py`,
`work/server_conditional_profile_full.sh`,
`work/average_profile_finals.py`.

Frozen SHA-256: decision
`3b746f039ac5d2b6cf8b1b9683d844632cc90b8879d178da09df8a7804e4300b`,
averager `c130d7ea354264dbf91c0c142b10b69388e1c7f9b029ab38d447aaa6641f12e7`,
server tail `81648f9658887a1866e7793065b6ab72977e0e6568146c1de3b471d1e9006d15`,
watcher `2e873e467c7d5dadf466e2b2b7c7bc20f69597fa320f605aa1820105f307af9f`.

### Relative-position TCN — frozen before matched-control result

Механизм: обычный pooled TCN получает calendar целевого окна, но не явную
позицию каждого дня внутри 409-дневной истории. Global mean/max и recent-14
pooling могут терять различие между одинаковым событием 10 и 300 дней назад.
К 17 raw channels добавлены только два label-free deterministic channels:
линейный относительный age `[-1,0]` и `exp(-age/30)`. Несуществующая левая
padding-часть зануляется. Width, blocks, loss, anchors, summaries, calendar и
все остальные параметры совпадают с exact control.

Seed `1310/2718` запускаются независимо, каждый с fold342 selection и fold378
untouched. Same-seed control уже стоит раньше в очереди. На обоих seed
обязательны: single independent gain `≥0.00012`, positive `≥90/96`, negative
weight `≤10%`, conditional gain сверх `w409c ≥0.00008` с тем же sign-gate и
untouched gain над same-seed control `≥0.00020`. Только simultaneous pass
разрешает два full-refit, fixed epoch/mix, стандартизацию и 50/50 seed average;
локальный watcher тогда строит единственный plain `127` probe. Тот же заранее
описанный diagnostic-tier (`0.00004/0.00002/75%`, holdout не хуже control более
чем на `0.00050`) может разрешить только измерительный probe, но не final
admission. Fail обоих tiers закрывает ветку без rescue или public tuning.

Generator/source control: `work/generate_position_trainer.py` hash
`145db81fbb347b4cbf7a36a133df04d70c2b9148178a44e63b85c6c456746e44`;
generated trainer
`edb36b26524369c0e8841e0990d4e20ae425c857f4edca4fdf3cb98d30327d65`;
decision `e9111ab1debaa3e0e2b2b41da25b5e4665dca57bb0c1a8981dd37b97da640023`;
server tail `b999f06ae16fb168d53c660858a5a9b2ce46189a01c53c4c91277f15d8f85982`;
watcher `a55ecd2fb7ccf4d684561527f6b53e88d4a7e27ffd7f6abac66f19c88c47253e`.
Local smoke-test completed one full train/predict step on 4096 users with
29 input channels, finite validation output and no shape error.

До любых position results также frozen morning stability tail: seed `31415`
всегда считает exact control, затем relative-position и marked-event с теми же
fold342/378 и 96-split audits. Он не выбирает ветку и не делает full/CSV:
это независимая проверка устойчивости двух ещё живых гипотез примерно до
10:00 МСК.
Implementation `work/server_morning_tail.sh`, hash
`f69ddfebcb5b6a4026109beb50b8787ef9c81d9e81f0e2032fac2bc343c3b85e`,
supervisor PID `430198`.

Seed31415 не может rescue-ить ветку, rejected по первым двум seed. Для уже
strict/diagnostic семьи он обязан пройти те же thresholds; только тогда утром
разрешается рекомендация ранее построенного probe. Consolidated report:
`work/seed31415_confirmation.json`; evaluator hash
`5d3c2aa2036bd2541c4a831620c3bdae73571ee2713b005d6324bb45f3c4e618`,
fetcher hash
`81868003dd4751623bdc3d32dd1c7477690fbe1991a76a324b78f55b4bbe23f9`.

Каждый созданный `124–127` probe до ручной отправки автоматически получает
machine-readable positive/strict+adaptive reading key через frozen
`work/freeze_profile_probe_gate.sh` (hash
`696a91c3501afccab87cb4b230976121a113934575ea35aaa61e8e55326e16a4`).
Gate включает весь measured block `83/85/86/89/92/102`, `lambda=.003`, двойную
empirical df-price и фиксированный adaptive cost `0.00004`. Поэтому даже
diagnostic-tier не допускает post-result придумывания порога.

### Multi-scale hidden decay pooling — frozen before night results

Frozen 26.08.2026 в `09:35 МСК`, когда SSH был недоступен и profile/position
decisions ещё не были просмотрены. Механизм: exact control после convolutional
trunk сохраняет global mean/max и recent-14, но теряет время возникновения
скрытого паттерна. Добавлены четыре детерминированных weighted averages hidden
states с decay `7/30/90/180` дней. Raw channels, calendar, summaries, loss,
width, blocks и anchors не меняются; это representation ablation, а не tuning.

Seed `1310/2718`, fold342 selection → fold378 untouched, same-seed controls и
strict/diagnostic gates точно совпадают с relative-position protocol. Только
two-seed pass разрешает full seed-average и один plain `128` probe от frozen
baseline; automatic submission отсутствует. Smoke-test на 512 users завершил
train/predict с finite output и корректной конфигурацией.

Frozen hashes: generator
`511ec989416acbcadbe7d92162e872452cbe3a0f30eb8f1de77649652a8cc8dd`,
trainer `dc6c683c5ee2fa49c2a11699079b5b8d5abfebdc03de7378b702e962791a9dcc`,
decision `a72bf5ec764d2d2a5e95dce2d1369e5aef81b29b3e6cbccc37023c423f0509ba`,
server `c3601d5d638d964b41a4847e95248b18d47c4e230c2f6d411431d1dee86544c9`.

Одна тяжёлая GPU-задача выполняется за раз:

GPU-цепочка остаётся последовательной: `event cadence → direct-only event →
regularity profile → matched control → marked-event profile`. После decisive direct-only holdout fail
pair-safe CPU diagnostic безопасно вынесен параллельно: GPU trainer использует
примерно одно CPU-ядро, а diagnostic — 24 из 32; RAM `≈9/125 GiB` до старта.

Pair-safe supervisor PID `353864` стартовал в `17:37:55 МСК`; GPU-run не
останавливался. Это избегает 8–10 GB локального memory pressure и использует
128 GB server RAM. Local fetch watcher session `2184`.

Conditional lagged-residual supervisor PID `353865` уже стоит после pair-safe;
он всегда пишет verdict, но создаёт plain CSV только при прохождении всех
preregistered trigger/promotion gates. Local fetch watcher session `40591`.

Фактический preregistered trigger по первым семи парам также провален:
`mean=−0.001280 < 0.003`, положительны 3/7 вместо 6/7. Поэтому conditional
builder корректно записал `promoted=false` и не создал ни кандидата, ни CSV;
authoritative verdict `work/lagged_residual_clean_candidate_meta.json`.

Чтобы сервер не простаивал при исчерпании интерактивной квоты, заморожен
осмысленный хвост: marked-event seed2718 → matched-control seed2718, каждый с fold342 selection и
fold378 untouched, без full/CSV. Handoff hashes:
`server_five_hour_tail.sh = f3b4d9a5eaf596b0d7872388d50b75f4aef16ac25cade64ce2c3277105aba5db`,
`server_five_hour_control_tail.sh = e48bb51087fd3e2a4279dd318ebf2d4dd3cb48593e4dcfa3894640c7fae1160a`.
После инфраструктурного fail из-за отсутствовавшей frozen trainer-копии
последовательность была полностью перезапущена в `22:38:39 МСК`; незавершённый
run не используется. Supervisor PIDs: control `407073`, marked `407074`, seed
profiles `430167`, seed control `430175`, audits `430181`. Отдельный frozen
conditional-full supervisor `430186` ждёт PID `430181` и не может стартовать
full до machine-readable strict/diagnostic pass. Position supervisor `430192`
стоит следом и использует уже завершённые exact controls.

Провал любого frozen gate закрывает ветку без full refit и без leaderboard.
Новые follow-up допускаются только по заранее сформулированному механизму, а не
для спасения результата post-hoc.

## Frozen same-season probe `121`

До возможного public-ответа зафиксирован `work/121_public_reading_key.json`.
`121_offline_gift_holdout.csv` — неизменяемая leaderboard-free модель формы,
обученная на user-holdout для окна 14.02–15.03.2025 и перенесённая на 2026 с
historical-only прогнозом уровня. Это отдельная сезонная гипотеза, а не lambda-
вариант. Ответ используется один раз для восстановления aggregate cross-moment;
promotion требует conditional gain `≥0.00012`, положительный вес, минимум
90/96 положительных independent splits и максимум 5/96 отрицательных весов.
Архитектуру, калибровку, vector и gates после ответа менять запрещено.

Результат: внутренний same-season audit был положителен даже против oracle
single (`mean gain=0.00030684`, 91/96, p05 `0.00000269`), однако public
`1.7621018713` показал полный провал year-over-year transfer. Frozen conditional
direction оказался неположительным; ветка закрыта без blend/retrain. Report:
`work/121_same_season_stability_audit.json`.
