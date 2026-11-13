# VisDrone-4NN: классификация объектов и оценка устойчивости

Подготовка гражданского VisDrone-DET для сравнительного исследования архитектур.
Реализованы подготовка данных и baseline fine-tuning четырёх pretrained-классификаторов.
Добавлена оценка FGSM и PGD на validation. Защиты и финальная оценка holdout
не реализуются на этом этапе.
Данные и веса сторонних проектов не включены; исходный датасет пользователь
получает у официального VisDrone с соблюдением условий использования.

## Данные

Ожидаемая структура (архивы распаковать заранее):

```text
VisDrone/
  VisDrone2019-DET-train/
    images/*.jpg
    annotations/*.txt
  VisDrone2019-DET-val/
    images/*.jpg
    annotations/*.txt
```

Каждая строка DET: `left,top,width,height,score,category,truncation,occlusion`.
Классы 1–10 отображаются в метки 0–9: pedestrian, people, bicycle, car, van,
truck, tricycle, awning-tricycle, bus, motor. Сохраняются все допустимые объекты,
включая перекрытые и усечённые. Score=0, ignored regions (0), others (11)
не становятся обучающими примерами. Наличие ignored region не удаляет другие
размеченные объекты изображения. Некорректные строки и рамки учитываются в
статистике пропусков. Частичные рамки ограничиваются границами изображения;
полностью внешние, нулевые и отрицательные рамки исключаются. EXIF-поворот
не применяется, чтобы сохранить координаты DET.

Отсутствие аннотаций или повреждённое изображение вызывает ошибку. Парсер
ожидает ровно 8 полей и конечные числовые значения. Crop создаётся при чтении,
а не сохраняется отдельным файлом. Пустые аннотации допустимы, но каждый итоговый
split должен содержать хотя бы один объект.

## Запуск этапа 1 — только для новой подготовки данных

**Для уже подготовленного `data-v1` этот раздел пропустите. Не запускайте `data.prepare` повторно.**

Python 3.10+; локально рекомендуется отдельное виртуальное окружение:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m data.prepare --root /path/to/VisDrone --output /path/to/new-results
```

Значения по умолчанию совпадают с `configs/data.json`; CLI предоставляет
`--train-split`, `--val-split`, `--holdout-fraction`, `--seed`.
Каталог output должен отсутствовать; при ошибке подготовки может остаться
частичный каталог, который нельзя считать готовым без manifest.json и statistics.json.
Исправив причину, выберите новый output.

В Colab откройте `notebooks/01_visdrone_data.ipynb`, подключите Drive, выполните
клонирование этого репозитория после сохранения изменений в GitHub, установку и тесты.
Укажите ROOT и новый OUTPUT. Для этого этапа достаточно CPU; T4 не требуется.
После перезапуска достаточно снова открыть checkout, установить зависимости и
загрузить сохранённые JSON: пересоздавать разбиение не нужно.

Выход: train/validation/holdout.json с относительными путями и рамками,
manifest.json с принадлежностью исходных изображений и SHA-256 аннотаций,
statistics.json со всеми 10 классами и причинами пропуска, случайные PNG-примеры.
Ноутбук дополнительно сохраняет commit и версии пакетов. SHA-256 не проверяет
содержимое изображений: для строгого архива сохраняйте также контрольные суммы
оригинальных архивов. Диапазоны зависимостей не являются lockfile; версии
проверенного запуска фиксируются в environment.txt.

## Протокол и ограничения

Official train остаётся train; размеченный official val детерминированно делится
на validation (80%) и финальный holdout (20%) по исходным изображениям. Все crop'ы
одного источника остаются вместе. Не используйте неразмеченный official test-dev
для расчёта метрик. Holdout нельзя использовать для выбора модели, защит или
бюджетов атак. Разбиение случайное, не стратифицированное; оно не гарантирует
наличие редких классов и не предотвращает утечку между соседними кадрами одной
сцены. Проверьте provenance и сходство сцен до фиксации протокола.

DataLoader воспроизводимо перемешивает данные и инициализирует worker seeds.
Проверочный preprocessing: RGB, bicubic resize 224×224, float32 CHW [0,1].
Он меняет пропорции crop. Это не финальный preprocessing pretrained-моделей:
для обучения используется отдельный transform из timm (см. этап 2 ниже).
Балансировка классов и аугментации на этом этапе не вводятся.

## Проверка настоящего датасета

- Проверьте источник, лицензию, версию архивов и их контрольные суммы.
- Сверьте число исходных изображений/аннотаций с официальным релизом.
- Изучите все причины пропусков: неожиданные malformed/invalid требуют разбора.
- Проверьте примеры всех классов, пограничные рамки, occlusion и truncation.
- Проверьте поддержку всех 10 классов в каждом split и дисбаланс классов.
- Проверьте отсутствие общих исходных ID и дубликатов/сходных сцен между частями.
- Выполните DataLoader-ячейку на реальных данных и сохраните артефакты протокола.

Автотесты используют только синтетические изображения; они не доказывают
корректность подготовки конкретного релиза VisDrone. Длительное обучение не запускается.

## Этап 2 — baseline-обучение

Используются существующие `train.json` (343 170 объектов / 6 471 снимок) и
`validation.json` (30 570 / 438) из `/content/drive/MyDrive/visdrone-results/data-v1/`.
Эти числа предоставлены пользователем, в среде разработки не перепроверялись.
Holdout (8 189 / 110) не открывается. Индексы не изменяются; bbox, line, image и
метки остаются в `CropRecord`, включая объекты размером менее 16 px.
Смена `--root` на локальную копию не меняет относительные пути индексов.

### Структура новых компонентов

```text
configs/baseline.json                # точные weight tags и общие defaults
training/config.py                  # проверка и сохранение параметров
training/models.py                  # реестр timm, одна модель и transforms
training/data.py                    # чтение индексов, subset, loaders
training/engine.py                  # один train epoch
training/validation.py              # loss, accuracy, macro/per-class F1
training/checkpoint.py              # атомарные best/last, RNG, restore
training/run.py                     # CLI и управление экспериментом
notebooks/02_baseline_training.ipynb # Drive → smoke → full → resume
tests/test_training.py              # синтетические проверки
```

### Выбранные модели и preprocessing

Зафиксирован `timm==1.0.22`. Полные идентификаторы проверяются по
`list_models(pretrained=True, include_tags=True)` и `get_pretrained_cfg`.
Отсутствующий вариант вызывает ошибку без подмены модели или случайной инициализации.

| CLI model | Полный timm ID | Предобучение | mean/std | crop_pct |
|---|---|---|---|---|
| `convnext_base` | `convnext_base.fb_in1k` | ImageNet-1k | ImageNet | 0.875 |
| `convnext_nano` | `convnext_nano.d1h_in1k` | ImageNet-1k | ImageNet | 0.95 |
| `vit_base` | `vit_base_patch16_224.augreg_in1k` | ImageNet-1k, AugReg | 0.5 / 0.5 | 0.9 |
| `tinyvit` | `tiny_vit_5m_224.dist_in22k_ft_in1k` | ImageNet-22k distillation → 1k | ImageNet | 0.95 |

ImageNet mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225).
Все входы RGB 224×224, bicubic, pretrained resize + center crop.
Параметры берутся программно через `resolve_model_data_config` и `create_transform`,
а не из таблицы. Используется существующий `VisDroneCrops(transform=...)`:
PIL crop → transform → tensor; прежний `ClassificationTransform` не применяется,
двойной нормализации нет. Train добавляет только horizontal flip p=0.5 перед тем же
preprocessing. RandomResizedCrop не используется, чтобы дополнительно не вырезать
малые цели. Center crop pretrained-конфигурации может обрезать края вытянутых crop:
проверьте визуализацию в ноутбуке. Рамки исходного объекта не меняются.

**Ограничение сравнения:** TinyViT-5M — гибридная архитектура с другим масштабом
и способом предобучения. Рецепты pretrained-весов различаются и у других моделей;
эффект архитектуры нельзя полностью отделить от предобучения. Не трактуйте
результаты как причинный эффект только CNN против Transformer.

### Точная последовательность в Google Colab

Сначала сохраните эти изменения в своём репозитории и откройте
`notebooks/02_baseline_training.ipynb`. Подключите T4 GPU и Google Drive.
Ноутбук не запускает подготовку данных и по умолчанию не копирует датасет.
Существующий checkout автоматически не обновляется: выберите нужный commit вручную.

Из корня checkout `/content/visdrone-4nn`:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m training.models
python -m training.run --model convnext_nano --smoke --device cuda
```

При требовании pip перезапустить runtime сделайте это до импорта torch/timm.
Torch и torchvision должны быть совместимой парой; не устанавливайте CPU-only
колёса в Colab. Требуется Python 3.10+, torch>=2.4 для `torch.amp.GradScaler`.
`training.models` без `--download` проверяет реестр без сетевой загрузки.
Для отдельной проверки загрузки **одной** модели:

```bash
python -m training.models --model tinyvit --download
```

Smoke использует одну эпоху и максимум 2 batch каждой части, выбираемых с seed;
максимум 16 train и 16 validation при batch_size=8. Это проверка работоспособности,
не качества. Посмотрите loss, число примеров и визуализацию реальных crop;
проверьте наличие best.pt/last.pt и их чтение через resume в ноутбуке.
Только затем запускайте полное обучение. Каждая следующая команда — отдельный
ручной запуск после завершения предыдущего и успешного smoke выбранной модели:

```bash
python -m training.run --model convnext_nano --device cuda
python -m training.run --model convnext_base --device cuda
python -m training.run --model vit_base --device cuda
python -m training.run --model tinyvit --device cuda
```

Для smoke другой модели замените имя в первой команде с `--smoke`.
Новый full всегда начинает с pretrained-весов, не с весов smoke.
В памяти находится одна модель. Общие defaults: batch 8, epochs 10, lr 1e-4,
weight decay 0.05, workers 2, seed 42, AMP на CUDA, unweighted CrossEntropyLoss,
AdamW (все параметры), CosineAnnealingLR без warmup, drop_last=False.
Best выбирается по максимальному validation macro-F1; при равенстве остаётся
более ранняя эпоха. Loss усредняется по объектам, а не по batch. Macro-F1 всегда
среднее по всем 10 классам; F1 отсутствующего класса = 0. Validation без аугментаций.
Дисбаланс классов не компенсируется в этом baseline, чтобы не вводить дополнительный
фактор. Train перемешивается, validation — нет.

Все пути по умолчанию уже заданы для Drive в `configs/baseline.json`.
Пример явного указания путей и предварительного подмножества:

```bash
python -m training.run --model convnext_nano --root /content/VisDrone --indices /content/drive/MyDrive/visdrone-results/data-v1 --output /content/drive/MyDrive/visdrone-results/baseline-v1 --batch-size 8 --epochs 3 --lr 0.0001 --workers 2 --train-size 4096 --seed 42 --device cuda
```

`--train-size` выбирает фиксированное случайное подмножество train без пересоздания
split; validation остаётся полной (кроме smoke). Это предварительный эксперимент,
который следует явно отличать от обучения на всех 343 170 объектах. Для сопоставления
моделей используйте одинаковые seed, subset, batch и остальные гиперпараметры.
На T4 сначала оцените время эпохи: полный train велик, Free Colab может оборвать
сеанс до завершения эпохи. При OOM уменьшите batch для всех сравниваемых моделей
и создайте новые запуски. Автоматического изменения batch или подмены модели нет.

### Результаты и восстановление

`RUN_DIR` печатается при запуске. Каждый новый эксперимент создаёт уникальный путь:
`baseline-v1/<model>/{smoke|full}-<UTC timestamp>-<uuid>/`.
Результаты и `data-v1` должны лежать в отдельных деревьях каталогов; вложенный или
родительский output запрещён. Ничего из `data-v1` не записывается.

В RUN_DIR сохраняются:

- `config.json`, `provenance.json`: гиперпараметры, полный model ID, commit/status,
  SHA-256 исполняемого кода и конфигов, версии основных пакетов, CUDA и GPU.
- `source.zip`, `environment.txt`: снимок собственного кода/конфигов, pip freeze.
- `data.json`: SHA-256 train/validation JSON, fingerprint выбранных записей,
  распределения классов, число объектов и снимков, resolved data config.
- `preprocessing.json`: параметры и текстовое описание transforms.
- `metrics.json`: поэпоховые train/validation loss, accuracy, macro-F1,
  per-class F1/support, learning rate и время эпохи.
- `best.pt`, `last.pt`: model, optimizer, scheduler, AMP scaler, RNG, epoch,
  best score, history, config, fingerprints и provenance.
- `summary.json`: итог выполненного запуска, выборочная метрика и признак smoke.

Пример resume (подставьте точный существующий RUN_DIR):

```bash
python -m training.run --resume /content/drive/MyDrive/visdrone-results/baseline-v1/convnext_nano/full-YYYYMMDDTHHMMSSZ-ID
```

При смене места хранения изображений дополнительно укажите `--root /content/VisDrone`.
Код, версии основных пакетов, индексы, выборка, preprocessing, тип устройства, AMP,
план epochs и гиперпараметры при resume проверяются. Их изменение требует нового
эксперимента; завершённый запуск при resume не добавляет эпохи.
После переустановки runtime восстановите recorded environment (совместимую CUDA-пару
torch/torchvision) и нужный commit/source snapshot. Checkpoint загружается без новой
загрузки pretrained-весов. Используйте только доверенные собственные checkpoint:
они содержат сериализованные Python/numpy RNG state.

Возобновление — **с границы эпохи**. Незавершённая эпоха повторяется целиком.
Seed порядка DataLoader/worker-аугментаций привязан к epoch; RNG и состояния
оптимизации также сохраняются. `last.pt` — источник истины для history;
`metrics.json` восстанавливается из него при resume. Запись каждого файла идёт
через временный файл и rename; транзакции сразу для best/last/metrics нет.
При обрыве во время записи предыдущий last обычно остаётся пригодным, но гарантии
fsync/транзакционности Google Drive FUSE не даются. Не запускайте два resume одного
RUN_DIR одновременно. Проверьте доступность файлов после повторного mount Drive.
До конца первой эпохи last ещё нет — такой запуск нужно начать заново.
Детерминированные алгоритмы включены; неподдерживаемая операция вызывает ошибку,
а не скрытое отключение детерминизма. Битовая идентичность разных GPU/драйверов
не гарантируется; содержимое изображений при смене root не хешируется.

Локальная копия `/content/VisDrone` опциональна: notebook копирует только изображения
train/val **лишь при явной установке `COPY_IMAGES_TO_LOCAL=True`**. По умолчанию
чтение идёт с Drive. Учтите место на диске и потерю локальной копии после обрыва.
Checkpoints и журналы по умолчанию всегда сохраняются на Drive.

### Проверки и ограничения этапа 2

Синтетические тесты: реальные четыре архитектуры и forward с `pretrained=False`,
форма batch/меток, одинарная нормализация, сохранение малых bbox, subset/smoke,
метрики с отсутствующими классами, train step, checkpoint с Adam/scheduler/RNG,
равенство следующего шага и интеграционное восстановление после искусственного
обрыва. Интеграционные тесты используют маленькую тестовую сеть.
Проверка настоящей pretrained-загрузки и CUDA AMP выполняется в Colab; результаты
синтетических тестов не являются обучением или оценкой на VisDrone.

На настоящем VisDrone в Colab необходимо:

1. Сверить `data.json` с числами объектов/снимков и классами этапа 1; для full без
   subset ожидаются 343 170 train и 30 570 validation. Проверить неизменность JSON.
2. Выполнить smoke отдельно для всех четырёх моделей: загрузка реальных весов,
   смена головы на 10 классов, finite loss и tensors, CUDA AMP, отсутствие OOM.
3. Просмотреть исходные и обработанные crop, особенно маленькие и вытянутые цели,
   метки и поддержку всех классов в полном validation.
4. Проверить, что scaler CUDA имеет сохранённое состояние, оптимизация меняет
   параметры; протестировать отключение/новый runtime и resume полного запуска
   после одной завершённой эпохи, восстановление LR/optimizer/history.
5. Проверить длительность эпохи, скорость Drive/локального I/O, место под два
   полных checkpoint на модель (включают Adam states), сохранность после mount.
6. Зафиксировать единый протокол перед сравнением; holdout оставить закрытым.

Длительное обучение, загрузка VisDrone, атаки, защиты и профилирование этапа 3
в среде разработки не выполняются. По данным этапа 1 медианный эквивалентный размер
26.1 px; resize до 224 не восстанавливает утраченную детализацию мелких целей.

### AMP: пропуски обновлений оптимизатора

`train.optimizer_steps` и `train.amp_skipped_steps` в поэпоховых метриках показывают
число выполненных и пропущенных обновлений. При обычном GradScaler уменьшение
scale после `update()` означает пропуск из-за inf/NaN градиентов; равный или
увеличившийся scale означает выполненный шаг. Возвращаемое значение AdamW
для этой проверки не используется. Если за эпоху выполнено ноль обновлений,
scheduler и learning rate не продвигаются; validation и checkpoint сохраняются.

Формат checkpoint сохранён. Для исходной версии этапа 2 предусмотрено узкое
исключение проверки кода по точным SHA-256 `engine.py`/`run.py`; остальные
fingerprints и версии должны совпасть. Переход фиксируется в provenance.
Старые строки history без новых счётчиков читаются без изменений. Уже выполненные
старым кодом шаги scheduler при resume не отменяются; изменённое поведение
применяется к следующим эпохам. Проверка прежнего строгого соответствия кода
остаётся в силе для всех прочих изменений.

## Экономное хранение для Google Colab

Поведение без новых флагов сохраняется: полный `best.pt` и `last.pt` в OUTPUT.
Дополнительно экспортируется `best_weights.pt`. Архитектуры, preprocessing,
оптимизация, AMP и метрики не меняются.

В экономном режиме используйте два отдельных дерева каталогов:

```bash
python -m training.run --model convnext_nano --smoke --device cuda --economical --work-dir /content/visdrone-work --output /content/drive/MyDrive/visdrone-results/baseline-v1
```

После проверки smoke запустите ту же команду без `--smoke`. Для остальных моделей
замените `--model` и сначала выполните их smoke. Параметры:

- `--economical`: не создавать полный best.pt, экспортировать лучшие веса/метаданные.
- `--work-dir`: обязательный локальный базовый каталог при `--economical`.
- `--output`: долговременное хранилище результатов (Drive); прежний default сохранён.

RUN_DIR печатается для локального каталога, DURABLE_DIR — для Drive.
Имена `<model>/<smoke|full>-<timestamp>-<uuid>` совпадают в обоих деревьях.
В локальном RUN_DIR остаётся полный **last.pt** (model/AdamW/scheduler/scaler/RNG,
config/history/data/provenance). На Drive автоматически резервируется last.pt;
best.pt не копируется.
При улучшении validation macro-F1 сохраняется `best_weights.pt`; на Drive также
копируются config.json, preprocessing.json, data.json, provenance.json, metrics.json,
summary.json (в конце), storage.json, source.zip и environment.txt. Никаких файлов
data-v1 и holdout этот механизм не читает/не меняет. Экспорт начальных метаданных
выполняется до обучения; метрик — после каждой завершённой эпохи.

`best_weights.pt` — обычный `state_dict` без optimizer, RNG и Python-объектов:
все floating tensors в FP32 на CPU, целочисленные buffers сохраняют dtype.
Модель в памяти не преобразуется в FP32 и не изменяется экспортом.
JSON сохраняет полный timm ID/10 классов, параметры normalization/resize,
fingerprints индексов и provenance. Лучшая эпоха определяется первым максимумом
validation macro-F1 в metrics.json; это не обязательно последняя эпоха.

Безопасная загрузка для последующей оценки (без pretrained-загрузки):

```python
import json
from pathlib import Path
from training.models import build_model, make_transforms
from training.checkpoint import load_weights
run = Path('/content/drive/MyDrive/visdrone-results/baseline-v1/MODEL/RUN_NAME')
cfg = json.loads((run / 'config.json').read_text())
model, _ = build_model(cfg['model_id'], pretrained=False)
load_weights(run / 'best_weights.pt', model)  # torch.load(weights_only=True), strict=True
model.eval()
data_config = json.loads((run / 'preprocessing.json').read_text())['data_config']
_, evaluation_transform = make_transforms(data_config)
```

Продолжение при живом локальном каталоге:

```bash
python -m training.run --resume /content/visdrone-work/convnext_nano/RUN_NAME
```

Флаги режима повторять не требуется: они читаются из config.json. Исходный
DURABLE_DIR и его storage.json должны оставаться доступны. Пути output и режим
не меняются через resume. Перенос всего локального RUN_DIR с сохранением файлов
допустим: --resume указывает его новое расположение. Для изображений по-прежнему
можно указать --root, не меняя индексы.

**При уничтожении runtime Colab** используйте проверенный резервный last.pt на
Drive через --restore-from (инструкция ниже). Резервирование выполняется после
каждой зафиксированной эпохи, не после batch. Старые эксперименты не удаляются.
До конца первой эпохи checkpoint для resume ещё нет.

Для старых запусков обычный --resume сохранён: старый config получает значения
новых полей по умолчанию. Разрешены точные fingerprints исходного этапа 2 и
исправления AMP; остальные проверки кода/версий/данных остаются. Если отсутствует
best_weights.pt и имеется доверенный best.pt, resume экспортирует именно его
исторические лучшие веса, не last. Формат полноценных checkpoint не изменён.
Загружайте full checkpoint только из собственных доверенных файлов
(`weights_only=False` необходим для RNG). Режим существующего запуска не меняется;
новый экономный эксперимент создаётся отдельной командой.

Все checkpoint/экспортные файлы записываются через уникальный временный файл и
rename. Новые каталоги создаются эксклюзивно. storage.json связывает локальный
и долговременный каталог идентификатором запуска; чужой/существующий каталог не
присваивается. Синхронизация использует явный список файлов и не переносит
произвольные .pt. Старые эксперименты не удаляются. При ошибке записи прежний файл
сохраняется, ошибка останавливает запуск; повторный resume повторяет экспорт.
Набор файлов не является общей транзакцией, а Drive FUSE не гарантирует fsync:
после сбоя завершите успешный resume/экспорт перед оценкой результатов. Нужны
место под временную копию заменяемого файла и отсутствие параллельных resume одного
RUN_DIR. Числа 57.1/171.4 МиБ для Nano предоставлены пользователем; в этой среде
размер на реальном обучении и работа Drive/T4 не измерялись.

### Согласованность при аварийном прерывании

`last.pt` теперь является точкой фиксации эпохи. Перед его атомарной заменой
кандидат лучшей модели записывается в локальный `best_pending.pt`. В last.pt
добавлено необязательное поле `best_selection`: номер лучшей эпохи (с единицы),
macro-F1 и SHA-256 канонического содержимого тензоров FP32/integer buffers.
Формат checkpoint остаётся version 1. Только после фиксации публикуются
best_weights.pt, best.pt (обычный режим), metrics.json и summary.json.
Первый максимум macro-F1 по зафиксированной истории остаётся правилом выбора.

При resume проверяются история и best_selection. Лучшие веса восстанавливаются
из подходящего best_weights.pt, staging-файла или копии на Drive по checksum;
если лучшая эпоха последняя, их можно восстановить из модели last.pt. В обычном
режиме доступен также best.pt. Если доказуемо соответствующие веса утрачены,
resume завершается явной ошибкой. Нефиксированный кандидат не меняет лучший score
и не публикуется вместо выбранной модели. Staging-файл не копируется на Drive;
локально это дополнительно один компактный набор весов. Существующие checkpoint
и каталоги автоматически не удаляются, staging заменяется только новым кандидатом.

После согласования history/summary/weights публикуется `commit.json`: номер
зафиксированной эпохи, best_selection и SHA-256 файлов weights/metrics/summary.
Он копируется на Drive последним. `training.consistency.verify_commit(Path(...))`
проверяет набор файлов; load_weights автоматически проверяет его при наличии.
Частичное обновление Drive обнаруживается по несовпадению checksum. После сбоя
выполните resume с локальным last.pt: публикация повторяется, метрики и summary
восстанавливаются из checkpoint даже для уже завершённого запуска. До успешного
согласования не используйте частично выгруженные файлы для оценки.

Старые full checkpoint без best_selection поддерживаются: выбранную модель
подтверждает last.pt, если лучшая эпоха последняя, иначе соответствующий best.pt.
Для старого экономного запуска, где лучшая эпоха более ранняя, одних компактных
весов без проверяемой связи с checkpoint недостаточно: требуется доверенный best.pt
или запуск завершится понятной ошибкой. Не угадывайте их принадлежность эпохе.
Точные source fingerprints PR #2 также разрешены для перехода на исправление.

Атомарность остаётся пофайловой, без гарантий fsync Google Drive FUSE; commit.json
позволяет обнаружить частичную публикацию, но не заменяет локальный полный resume.
При уничтожении Colab runtime используйте автоматическую полную резервную копию
через --restore-from. При её отсутствии/повреждении полного resume нет. Не запускайте несколько процессов
resume одного RUN_DIR одновременно.


## Что делать, если Colab runtime уничтожен

В экономном режиме после **каждой зафиксированной эпохи** полный локальный last.pt
автоматически резервируется в DURABLE_DIR/last.pt. Это происходит непосредственно
после локального commit, до публикации лучших весов и остальных результатов.
В обычном режиме полный last.pt уже сохраняется прямо в OUTPUT; отдельная
резервная выгрузка и --restore-from предназначены для экономного режима.

1. Подключите Google Drive в новом Colab:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Восстановите checkout кода и совместимое окружение torch/torchvision/timm,
   выберите T4 для прежнего CUDA-запуска. Из корня проекта выполните:

   ```bash
   python -m training.run --restore-from /content/drive/MyDrive/visdrone-results/baseline-v1/convnext_nano/RUN_NAME
   ```

   Команда проверит backup.json, SHA-256, размер/формат, run_id, эпоху и metadata,
   затем создаст новый локальный RUN_DIR в сохранённом work_dir. Существующий
   каталог не перезаписывается. Для нового базового пути используйте
   `--work-dir /content/visdrone-recovered`. При ошибке частичная локальная папка
   остаётся для проверки; повторите восстановление в другом work-dir.

3. Проверьте RESTORED_RUN_DIR и напечатанную команду продолжения:

   ```bash
   python -m training.run --resume /content/visdrone-work/convnext_nano/RUN_NAME
   ```

   При необходимости добавьте --root с текущим расположением тех же изображений.
   Восстанавливаются AdamW, scheduler, GradScaler и RNG. restore-from сам не
   обучает модель. reconcile() восстанавливает best/metrics/summary из полного
   checkpoint, даже если их выгрузка до сброса runtime была прервана. При resume
   публикация Drive завершается до следующей эпохи.

Сохраняются все завершённые **и успешно зарезервированные** эпохи; незавершённая
эпоха или локально зафиксированная эпоха, чья выгрузка не завершилась до уничтожения
runtime, теряется. Это резервирование на границе эпохи, не после каждого batch.
Если локальный last.pt ещё доступен после ошибки Drive, повторный --resume сначала
резервирует именно его; повторного обучения зафиксированной эпохи нет.

### Формат, атомарность и ограничения резервирования

Одна актуальная полная копия last.pt на запуск; старые эпохи не архивируются.
backup.json содержит run_id, completed_epoch (с единицы), sha256, file_size_bytes,
checkpoint_format_version, timestamp_utc и hashes статических config/data/preprocessing.
Новый checkpoint дополнительно содержит run_id; старые checkpoint без поля
проверяются по владельцу локального каталога и конфигурации/данным, затем их
принадлежность связывается проверенным манифестом.

Копия пишется во временный файл в том же DURABLE_DIR. До rename проверяются
SHA-256 относительно локального файла, чтение доверенного полного checkpoint и
номер эпохи. Перед публикацией записывается маленький backup_pending.json с
проверенным hash и принадлежностью запуску. После атомарной замены last.pt
проверяется опубликованный hash, обновляется backup.json и hash проверяется вновь.
Если процесс погиб между rename и манифестом, verify_backup/restore-from использует
совпадающий по hash проверенный pending и восстанавливает backup.json. По одному
лишь факту успешного torch.load неподтверждённый файл не принимается.

На Drive постоянно добавляется **один полный last.pt** плюс два маленьких JSON;
во время обновления нужно дополнительное свободное место ещё под один полный
checkpoint. Старые измерения пользователя для Nano: примерно 171.4 МиБ для full
и 57.1 МиБ для compact (не проверены здесь на реальном обучении). Экономный режим
по-прежнему не копирует best.pt и best_pending.pt. После завершения пользователь
может вручную удалить полную резервную копию, сохранив компактные веса/метаданные;
полный resume тогда будет недоступен до создания новой копии из локального last.
Никакого автоматического удаления завершённых копий/экспериментов нет.

При ошибке staging, неверном hash/эпохе или заполненном Drive прежний опубликованный
last.pt остаётся нетронутым; локальный last.pt не удаляется, обучение останавливается.
Освободите место/восстановите доступ и выполните --resume. Временный файл текущей
операции очищается при обычной ошибке; после SIGKILL может остаться orphan staging,
который не принимается за checkpoint и может быть удалён пользователем вручную.
Файлы best_weights/commit/metrics не входят в backup.json: их согласует существующий
reconcile() и commit.json, без второй системы выбора лучшей модели.

Если last.pt отсутствует, повреждён или его hash не совпадает ни с backup.json,
ни с проверенным pending, восстановление останавливается с понятной ошибкой.
Если утрачен источник лучших весов более ранней эпохи, reconcile также откажется
угадывать. Используйте собственную другую проверенную копию; компактных весов
недостаточно для полноценного resume. Старые экономные запуски без backup.json
создадут резервную копию при --resume с сохранившимся локальным last.pt.

Full checkpoints загружаются только из доверенных собственных файлов
(weights_only=False из-за RNG). SHA-256 обнаруживает повреждение и частичные
записи, но не является подписью от злоумышленника. Drive FUSE не гарантирует
fsync или транзакцию для всех файлов: сохранность следует проверить после mount.
Не запускайте параллельные resume одного RUN_DIR. Настоящие T4/Drive и сброс
Colab следует проверить коротким запуском пользователя; разработка использует
только синтетические CPU-тесты.

## FGSM и PGD — validation, baseline v1 и отдельный Base v2

Ветка реализации основана на согласованном commit
`57395eefc0df3a0bb57b20ca7a918c2ac28d0cb7`. Код обучения, presets, архитектуры,
checkpoints, индексы и прежние результаты не изменяются. Ни holdout, ни train JSON
оценщик не открывает. Square/Patch, transferability и защиты не реализованы.

### Пространство атак и предобработка

1. `VisDroneCrops` читает PIL RGB crop по существующей рамке validation.json.
2. Геометрия **точно из существующего evaluation-transform** timm: resize/center
   crop, bicubic, затем преобразование в float32 RGB 224×224 в [0,1].
3. FGSM/PGD меняет этот RGB-тензор; epsilon относится к его пиксельной шкале.
4. `PixelModel` дифференцируемо вычисляет `(rgb - mean) / std`, затем классификатор.

Из timm Compose удаляется ровно один последний Normalize. Если структура иная,
код выдаёт ошибку вместо скрытой подмены. Сохранённый data_config сверяется с
resolve_model_data_config и метаданными обучающего запуска. Ни тренировочных
аугментаций, ни дополнительного resize/нормализации внутри атак нет.

| Модель | Вход | crop_pct | mean/std |
|---|---|---|---|
| ConvNeXt-Base | 224×224 | 0.875 | ImageNet |
| ConvNeXt-Nano | 224×224 | 0.95 | ImageNet |
| TinyViT-5M | 224×224 | 0.95 | ImageNet |
| ViT-B/16 | 224×224 | 0.9 | 0.5 / 0.5 |

Общая размерность **не означает одинаковый геометрический вид объекта**: различные
crop_pct могут удалять разную часть краёв вытянутых crop. Это сохраняет обученный
preprocessing, но является фактором сравнения. Возмущение задаётся **после**
геометрии, не на исходном снимке/рамке; 1/255 относится к float RGB, без повторной
8-bit квантизации перед inference. Resize не восстанавливает детали малых целей.
Предобучение (особенно distillation ImageNet-22k у TinyViT) также различается.

### Единый протокол и API

`attacks.core.generate(model, images, labels, AttackConfig(...))` — общий интерфейс.
Модель должна принимать RGB (обычно через PixelModel), images — FP32 NCHW, labels —
истинные int64 метки. Атаки нетаргетированные, с CrossEntropyLoss:

- FGSM: один шаг alpha=epsilon, без random start.
- PGD: 20 шагов alpha=epsilon/4, один uniform random start в [-epsilon,+epsilon]
  с clipping в [0,1], проекция к **исходному** RGB после каждого шага, без рестартов.
- CLI разрешает epsilon `0 1/255 2/255 4/255 8/255`. Ноль — контроль, возвращает
  исходные пиксели без градиентных шагов; nominal параметры сохраняются с control=true.

Linf и диапазон проверяются после каждого шага и независимо в оценщике.
Численная погрешность Linf допускается до 1e-6, фактический Linf сохраняется.
Атаки и clean inference выполняются FP32 без AMP; TF32 отключён и включены
детерминированные алгоритмы через существующий seed_everything. Операции без
детерминированной реализации могут вызвать явную ошибку. Это не обещание битовой
идентичности разных GPU/драйверов/версий библиотек или batch size.

`autograd.grad` запрашивает только градиент по RGB: optimizer не создаётся, веса,
requires_grad и прежние parameter.grad не меняются. Для атаки временно включается
eval; исходные training-флаги всех submodules восстанавливаются даже при ошибке.
Случайный старт PGD имеет отдельный torch CPU Generator на объект, seed вычисляется
из SHA-256([base_seed, sample_id]); он не зависит от модели/epsilon/порядка batch
и не расходует глобальное RNG. Специально не выбирается лучший iterate по loss:
оценивается последний из 20 шагов основного протокола.

### Готовые команды будущего пилота Kaggle

Из checkout проекта после установки `requirements.txt` задайте реальные пути
к своим уже подключённым input datasets (имена ниже — примеры):

```bash
MODEL_RUN=/kaggle/input/visdrone-baselines/convnext_nano/RUN_NAME
VISDRONE_ROOT=/kaggle/input/visdrone2019-det/VisDrone
VALIDATION_INDEX=/kaggle/input/visdrone-data-v1/data-v1/validation.json
ATTACK_OUTPUT=/kaggle/working/attack-results

python -m attacks.evaluation --model-run "$MODEL_RUN" --checkpoint "$MODEL_RUN/best_weights.pt" --variant baseline-v1 --root "$VISDRONE_ROOT" --validation-index "$VALIDATION_INDEX" --output "$ATTACK_OUTPUT" --attack fgsm --epsilon 0 1/255 2/255 4/255 8/255 --subset-size 256 --seed 42 --batch-size 8 --device cuda --save-examples 4

python -m attacks.evaluation --model-run "$MODEL_RUN" --checkpoint "$MODEL_RUN/best_weights.pt" --variant baseline-v1 --root "$VISDRONE_ROOT" --validation-index "$VALIDATION_INDEX" --output "$ATTACK_OUTPUT" --attack pgd --epsilon 0 1/255 2/255 4/255 8/255 --subset-size 256 --seed 42 --batch-size 8 --device cuda
```

Это команды для **ручного будущего пилота**, не запускаемые тестами. Можно выбрать
512 вместо 256. Без --subset-size используется весь frozen validation; ожидаемые
30 570 объектов предоставлены пользователем, здесь на реальном VisDrone не проверены.
CLI одной командой последовательно оценивает один checkpoint на списке epsilon,
создавая отдельный уникальный run для каждого значения. В памяти одна архитектура.
Для остальных baseline измените MODEL_RUN на нужный запуск Nano/Base/TinyViT/ViT.

Base v2 на 50 000 crop с lr=3e-5 запускайте с **--variant convnext-base-v2** и его
собственным MODEL_RUN/checkpoint. Загрузчик не позволит пометить этот вариант как
baseline-v1 или подменить им основной четырёхмодельный baseline. Variant сохраняется
в конфигурации и каждой строке CSV. Smoke-trained checkpoints отклоняются.

Для полного доверенного checkpoint замените параметры загрузки на:

```text
--checkpoint /path/to/best.pt --checkpoint-format full --trusted-full-checkpoint
```

Без явного флага доверия pickle/RNG из full checkpoint не загружается. Для compact
используется weights_only=True с обязательной сверкой SHA-256 против
MODEL_RUN/commit.json; манифест из каталога весов не подменяет эту проверку. MODEL_RUN должен содержать согласованные config.json,
preprocessing.json, data.json, provenance.json. При неполной публикации сначала
исправьте/восстановите **свой обучающий запуск существующими средствами**, затем
передайте его оценщику. Сам оценщик ничего в MODEL_RUN не пишет и не скачивает
pretrained-веса; strict state_dict loading не оставляет случайной головы.

### Идентификаторы, результаты и прерывание

Выборка: sorted(random.Random(seed).sample(...)) позиций исходного validation.json.
Никакого нового split или фильтра по размеру/классу; bbox и line сохраняются.
sample_id = `index_sha256:original_zero_based_row:record_sha256`; он устойчив для
того же frozen файла и связывает объект с изображением, рамкой, классом и строкой.
Изменение байтов JSON меняет index hash и пространство идентификаторов. Хеш
подмножества включает выбранные записи, их позиции и attack seeds.

Новый `OUTPUT/<uuid>/` создаётся эксклюзивно; перезапись и resume атак отсутствуют.
Запуски разных моделей/variant/checkpoint/epsilon не переиспользуются. OUTPUT
должен находиться отдельно от изображений, индексов и model-run. Файлы:

- `config.json`: параметры атаки/FP32, variant, полный model ID, checkpoint path/SHA,
  training metadata, preprocessing, hash индекса и подмножества, размер batch,
  seed, Git commit/status/diff hash, hashes кода attacks/, Python/torch/torchvision/
  timm/CUDA. Несохранённый код дополнительно идентифицируется собственными hashes.
- `subset.json`: выбранные позиции, sample_id, image, bbox, line, класс, attack_seed
  и правило формирования. Ни один исходный индекс не записывается.
- `predictions/batch-*.csv`: атомарные порции с идентичностью объекта/модели/весов,
  true/clean/adversarial class, clean_correct, attack_success, epsilon, base/object
  seed, параметрами атаки и фактическим Linf.
- `metrics.json`: агрегаты для завершённых порций. Accuracy делится на **все**
  объекты. ASR — только на clean-correct; при нулевом знаменателе JSON null и
  asr_defined=false. В CSV attack_success пуст для изначально ошибочных объектов.
  Macro-F1 и per-class F1 используют все 10 классов, нулевой знаменатель F1 → 0
  по существующему ClassificationMetrics. Есть clean_correct_per_class и F1 drop.
- `status.json`: running/completed/failed, число записанных объектов, UTC начало/
  завершение, причина ошибки. `diagnostics.jsonl`: нулевые/нечисловые градиенты,
  нечисловые logits/loss с batch/sample IDs. Nonfinite останавливает запуск.
- Опционально `examples/` (0 по умолчанию, максимум 16): clean/adv RGB PNG и signed
  FP32 `*-delta.pt` (читается weights_only=True), JSON sample_id/epsilon. PNG только
  визуализация с 8-bit округлением; точное delta сохранено тензором. Все изображения
  не сохраняются.

Порция CSV сначала записывается во временный файл и атомарно публикуется, затем
обновляются metrics/status. При обрыве возможен отстающий metrics.json, но уже
опубликованные CSV остаются пригодны для анализа. SIGKILL может оставить .tmp,
который читатель не рассматривает как порцию. Повторный запуск всегда новый uuid;
автоматического продолжения/дублирования строк в прежнем run нет. Пример чтения:

```python
from pathlib import Path
from attacks.results import read_predictions, summarize
run = Path('/kaggle/working/attack-results/RUN_UUID')
rows = read_predictions(run)  # Проверка модели/checkpoint/epsilon и уникальности ID.
metrics = summarize(rows)    # Истина для опубликованного префикса при прерывании.
```

Результаты частичного подмножества должны быть явно обозначены как partial,
не смешивайте их с complete. Файловая атомарность не гарантирует fsync/сохранность
временного Kaggle-диска после уничтожения runtime; для длительного хранения
пользователь отдельно сохраняет новый каталог результатов. Никакой автоматической
публикации в checkpoint datasets или изменения старых экспериментальных файлов нет.

### Проверки и ограничения до массового запуска

CPU-тесты проверяют аналитический знак FGSM/рост CE, каждый шаг проекции PGD,
число итераций и реальное изменение iterate, RNG, веса/градиенты/режим модели,
нормализацию всех четырёх timm configurations без двойного применения и градиент
до RGB, точные метрики/нулевой ASR-знаменатель, строки/агрегаты, загрузку strict
compact/full без pretrained, уникальность run и сохранение префикса после сбоя.
Синтетический CLI smoke покрывает несколько epsilon и основной 20-step PGD.

Перед реальным пилотом проверьте пути/идентичность четырёх baseline и отдельного
Base v2, training provenance, число validation-объектов, визуальную геометрию crop,
совпадение epsilon=0 с обычным inference, ненулевые конечные градиенты всех четырёх
реальных обученных моделей, T4/Kaggle память/скорость и достаточность выбранного
batch. Для сравнения сохраняйте одинаковые выбранные sample IDs и base seed.
В unit-тестах реальные обученные веса/VisDrone не использовались; инференс с
настоящими checkpoint и атаки на GPU здесь не выполнялись. Хеши исходных изображений
не считаются: неизменность image datasets необходимо обеспечить отдельно.

PGD-20 с одним стартом не доказывает отсутствие более сильных атак. PGD не обязана
превосходить FGSM на каждой отдельной выборке, как и рост loss не доказывает ASR:
успех определяется только ошибкой на изначально верно распознанном объекте.
Не выбирайте бюджеты/защиты на holdout; он остаётся закрытым. Этот этап завершается
FGSM/PGD validation-инфраструктурой, без автоматического перехода к будущим атакам.

Проверка этой реализации в Codex: `python -m pytest -q` — **92 passed, 1 skipped**
(CUDA недоступна), одно существующее предупреждение torch.jit.interface.
Отдельный CPU smoke: шесть синтетических crop, FGSM eps=0/4/255 и PGD-20 eps=4/255;
все три запуска completed, max Linf для ненулевого epsilon = 0.0156862735748291
при бюджете 4/255 = 0.01568627450980392. Это проверка конвейера, не результат
устойчивости обученных VisDrone-моделей. `git diff --check` и проверка CLI пройдены.

### Проверки identity и целостности перед пилотом

Для компактных весов обязателен `MODEL_RUN/commit.json`: SHA-256 переданного файла
должен совпадать с `sha256.best_weights.pt` именно этого запуска. Раздельное
расположение metadata и weights поддерживается только при таком совпадении;
соседний манифест чужого checkpoint не заменяет связь с MODEL_RUN. Отсутствие
манифеста — ошибка, а не переход к проверке только архитектуры. Манифест и его
hash сохраняются в provenance оценки. Для доверенного полного checkpoint также
проверяется hash содержимого тензоров против best_selection.sha256 MODEL_RUN:
он должен содержать именно опубликованную лучшую модель, а не произвольный last.

`read_predictions` сверяет subset hash/size с config.json, каждую строку CSV с
subset.json (sample_id, исходная позиция, image, box, line, label, attack_seed),
последовательность/размер CSV-порций и отсутствие дубликатов. completed требует
полного покрытия и точного совпадения сохранённых метрик с пересчитанными.
Удалённые chunks, чужие метки и подменённые ID отклоняются.

Для running/failed по умолчанию также проверяются saved metrics/status. Только
если обрыв произошёл между публикацией CSV и записью агрегатов, можно явно вызвать
`read_predictions(run, allow_stale_partial=True)` и затем summarize(rows): читатель
выдаст предупреждение об отстающих агрегатах. Этот режим не разрешает нарушенную
идентичность, пропуски внутренних chunks или неполные результаты completed и не
изменяет файлы. Результат остаётся partial. SHA-256 даёт проверку связи с доверенным
манифестом, а не цифровую подпись против согласованной подмены всех входных файлов.

После доработок review PR #4 полный CPU suite: **109 passed, 1 skipped**
(существующий CUDA-тест), git diff --check пройден. Проверены подмена compact/full
checkpoint той же архитектуры, раздельные metadata/weights, чужой соседний manifest,
подменённые поля объектов, пропавшие/обрезанные chunks, неверные metrics/status,
изменённый subset и явное чтение stale partial. Протокол FGSM/PGD не менялся.
