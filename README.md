# VisDrone-4NN: классификация объектов — этапы 1 и 2

Подготовка гражданского VisDrone-DET для сравнительного исследования архитектур.
Реализованы подготовка данных и baseline fine-tuning четырёх pretrained-классификаторов.
Атаки, защиты и финальная оценка holdout не реализуются на этом этапе.
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
