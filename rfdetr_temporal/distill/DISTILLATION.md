# Отчёт: Реализация дистилляции в Temporal RF-DETR

## 1. Обзор архитектуры

Система построена по схеме **KD-DETR** адаптированной к двустадийному (two-stage) DETR-детектору на базе RF-DETR. Дистилляция работает в двух режимах одновременно: **Specific Sampling** (слоты = обученные queries учителя) и **General Sampling** (случайные queries, re-drawn каждый шаг).

```
Учитель: RF-DETR-Large @ 704px (2D, замороженный)
Студент: RF-DETR-Small @ 384px (2D backbone + Temporal Fusion @ T=5)

Три форварда на каждый шаг:
  Branch 1 ── student-queries  ──► L_det   (Hungarian, GT)
  Branch 2 ── teacher-queries  ──► L_spec  (slot-aligned KL+L1+GIoU)
  Branch 3 ── random-queries   ──► L_gen   (slot-aligned KL+L1+GIoU)

  L_total = L_det + λ·L_spec + λ·μ·L_gen
  λ = distill_loss_weight = 1.0
  μ = distill_general_loss_weight = 0.5
```

---

## 2. Участники

### 2.1 Учитель — `FrozenRFDETRTeacher` (`distill/teacher.py`)

| Атрибут | Значение |
|---|---|
| Архитектура | RF-DETR-Large |
| Разрешение входа | 704×704 |
| `num_queries` | 300 |
| `group_detr` | 1 (inference slice) |
| `num_classes` | 1 (стеноз) |
| Чекпоинт | `rfdetr_runs/rfdetr_large_arcade2x_704_reg/checkpoint_best_total.pth` |
| Режим | `eval()`, `requires_grad=False` у всех параметров |
| Переопределение `.train()` | всегда возвращает `super().train(False)` |

Учитель хранит буферы:
- `refpoint_embed_weight` — (300, 4) первые Q строк из `lwdetr.refpoint_embed.weight`
- `query_feat_weight` — (300, 256) первые Q строк из `lwdetr.query_feat.weight`

Эти тензоры через `register_teacher_queries()` копируются в буферы студента (не параметры — не обучаются).

### 2.2 Студент — `TemporalRFDETR` (`model.py`)

RF-DETR-Small @ 384px + поверх него — `TemporalFusion` (cross-attention между кадрами окна T=5). Поддерживает три `query_mode`:

| `query_mode` | Что используется как queries | `group_detr` |
|---|---|---|
| `"student"` | `model.refpoint_embed`, `model.query_feat` (learnable) | 13 (training) / 1 (eval) |
| `"teacher"` | `model.teacher_refpoint`, `model.teacher_query_feat` (buffers) | 1 |
| `"general"` | внешний dict `{refpoint, query_feat}` (random, per-step) | 1 |

---

## 3. Слот-выравнивание (KD-DETR Slot Alignment)

### Проблема двустадийного детектора

В two-stage RF-DETR (`Transformer.forward`):
1. Encoder выдаёт `memory` из признаков.
2. Two-stage блок вычисляет `refpoint_embed_ts` — топ-k предложений из энкодера.
3. Финальный `refpoints_unsigmoid` = learned_offset(query_feat[i]) + refpoint_embed_ts[i].

Если студенту просто подать teacher-queries (`query_mode='teacher'`), его encoder всё равно вычислит **свои** `refpoint_embed_ts` из LR-признаков и смешает их с teacher-learned offset. Слот `i` студента будет нацелен на `student_LR_topk[i] + δ(teacher_learned[i])` — геометрически НЕ совпадает с учителем.

### Решение: pre-hook инъекция

**Учитель** ставит `forward_pre_hook` на `lwdetr.transformer.decoder`:

```python
def _decoder_pre_hook(_module, args, kwargs):
    self._captured_decoder_inputs["tgt"] = args[0].detach()       # (B, Q, D)
    self._captured_decoder_inputs["refpoints"] = (
        kwargs["refpoints_unsigmoid"].detach()                     # (B, Q, 4)
    )
    return None  # не модифицирует
```

Это захватывает **финальные** входы в decoder — тензоры уже после топ-k отбора + learned offset учителя на HR. Они возвращаются в `teacher_out` как `decoder_tgt` и `decoder_refpoints`.

**Студент** ставит `forward_pre_hook` на `self.transformer.decoder`:

```python
def _decoder_inject_pre_hook(_module, args, kwargs):
    inj = self._inject_decoder_inputs   # установлен перед вызовом
    if inj is None:
        return None
    new_args = (inj["tgt"], *args[1:])
    new_kwargs = dict(kwargs)
    new_kwargs["refpoints_unsigmoid"] = inj["refpoints"]
    return new_args, new_kwargs         # перехватывает
```

Когда `decoder_inputs` передан в `model.forward(...)`, хук **полностью заменяет** `tgt` и `refpoints_unsigmoid` перед тем, как decoder начнёт работать. Backbone + encoder + two-stage block студента всё равно работают (для получения `memory`), но геометрия слотов полностью определяется учителем. После вызова `_inject_decoder_inputs = None` (finally-блок).

**Результат:** студент разглядывает `memory` через deformable attention **в тех же пространственных точках**, что и учитель — slot `i` у студента и slot `i` у учителя описывают одну и ту же область.

---

## 4. Три ветки на шаг обучения

### Branch 1 — Detection

```python
outputs = model(images, query_mode="student")
loss_dict = criterion(outputs, centre_targets)  # SetCriterion + Hungarian
loss = Σ_k weight_dict[k] * loss_dict[k]
```

- Обычный supervised detection loss.
- Обучает `refpoint_embed`, `query_feat`, `temporal_fusions`, `transformer.*`.
- `group_detr=13` в train-режиме.

### Branch 2 — KD Specific Sampling

```python
with torch.no_grad():
    teacher_out_spec = teacher(teacher_centre)        # HR 704px forward

student_kd_spec = model(
    images,
    query_mode="teacher",
    decoder_inputs={
        "tgt":       teacher_out_spec["decoder_tgt"],        # (B, 300, 256)
        "refpoints": teacher_out_spec["decoder_refpoints"],  # (B, 300, 4)
    },
)
distill_spec = distillation_loss(student_kd_spec, teacher_out_spec, cfg)
loss += λ * distill_spec["loss_distill"]
```

- 300 фиксированных слотов — обученные queries учителя.
- Форвард студента + инъекция → decoder студента видит teacher-рефточки.
- **НЕ обучает** `refpoint_embed`/`query_feat` (буферы).
- Обучает `temporal_fusions`, `transformer.decoder` через distillation loss.

### Branch 3 — KD General Sampling

```python
gen_q = model.sample_general_queries(Q_g=100, device, dtype)
# gen_q = {"refpoint": (100, 4) U[0,1], "query_feat": (100, 256) N(0, 0.02)}

with torch.no_grad():
    teacher_out_gen = teacher.forward_general(
        teacher_centre, gen_q["refpoint"], gen_q["query_feat"],
        min_weight=0.1,
    )

student_kd_gen = model(
    images, query_mode="general", general_queries=gen_q,
    decoder_inputs={
        "tgt":       teacher_out_gen["decoder_tgt"],         # (B, 100, 256)
        "refpoints": teacher_out_gen["decoder_refpoints"],   # (B, 100, 4)
    },
)
distill_gen = distillation_loss(student_kd_gen, teacher_out_gen, cfg)
loss += λ·μ * distill_gen["loss_distill"]
```

- 100 случайных queries на каждый шаг (разные пространственные позиции).
- Учитель прогоняет те же queries через свой decoder (своп `refpoint_embed`/`query_feat` в `try/finally`).
- Покрывает области фона, которые specific sampling не затрагивает.
- `min_weight=0.1` — флор на per-query вес, чтобы фоновые запросы тоже давали градиент.

---

## 5. Функция потерь дистилляции (`distill/losses.py`)

Для каждого слота `i` вычисляется вес:

$$w_i = \max_k \; \sigma(\text{teacher\_logit}_{i,k})$$

Это максимальная foreground-уверенность учителя по слоту. Веса нормируют потерю: потери на объектных слотах вносят больший вклад, чем фоновые.

**L_kl** — Bernoulli KL между teacher и student:

$$L_{kl} = \frac{\sum_i w_i \cdot \text{KL}(p^t_i \| p^s_i)}{\sum_i w_i}$$

где $\text{KL}(p \| q) = p \ln\frac{p}{q} + (1-p)\ln\frac{1-p}{1-q}$, умноженный на $T^2$ (конвенция Hinton).

Если у студента `K_s` ≠ `K_t` у учителя — обе стороны сворачиваются к единственной foreground-вероятности через `amax`.

**L_l1** — L1 на normalised cxcywh боксах:

$$L_{l1} = \frac{\sum_i w_i \cdot \|b^s_i - b^t_i\|_1}{\sum_i w_i}$$

**L_giou** — element-wise 1−GIoU:

$$L_{giou} = \frac{\sum_i w_i \cdot (1 - \text{GIoU}(b^s_i, b^t_i))}{\sum_i w_i}$$

Итоговая дистилляционная потеря:

$$L_{distill} = \underbrace{2.0}_{kl} \cdot L_{kl} + \underbrace{5.0}_{l1} \cdot L_{l1} + \underbrace{2.0}_{giou} \cdot L_{giou}$$

---

## 6. Датасет и препроцессинг (`dataset.py`)

При `with_teacher_frame=True` `TemporalStenosisDataset` разделяет аугментацию на два пайплайна:

```
geom_aug  = HFlip / VFlip / Rotate / Affine  ← ReplayCompose
photo_aug = Blur / Brightness / Contrast / CLAHE  ← per-frame, только студент
```

1. `frame_0` прогоняется через `geom_aug` → сохраняется `saved_replay`.
2. `frame_1..T-1` — `ReplayCompose.replay(saved_replay, ...)`.
3. Teacher (centre frame @ 704px) — тот же `ReplayCompose.replay(saved_replay, ...)` + force-resize обратно в 704×704 если аугментация изменила canvas.
4. Студентские кадры дополнительно получают `photo_aug` (per-frame, случайный seed).

Важно: геометрические операции (flip/rotate) применяются **одинаково** к студентским и teacher-кадрам, что обеспечивает согласованность аннотаций.

---

## 7. Изоляция градиентов (подтверждено аудитом)

| Параметр | L_det | L_spec | L_gen |
|---|---|---|---|
| `student.refpoint_embed` | ✅ 6.65 | `None` ✅ | `None` ✅ |
| `student.query_feat` | ✅ 1.14 | `None` ✅ | `None` ✅ |
| `temporal_fusions` | ✅ | ✅ | ✅ |
| `transformer.decoder` | ✅ | ✅ | ✅ |
| `teacher.*` | — | 0 ✅ | 0 ✅ |

Студент-queries (`refpoint_embed`/`query_feat`) обучаются **только** через Branch 1 (detection). Дистилляционные ветки обучают **shared memory** — temporal fusion и decoder — чтобы студент научился генерировать хороший feature map, на котором decoder (в том числе в detection-режиме) работает лучше.

---

## 8. Гиперпараметры

| Параметр | Значение | Смысл |
|---|---|---|
| `distill_loss_weight` | 1.0 | λ для L_spec |
| `distill_general_loss_weight` | 0.5 | μ, множитель для L_gen |
| `distill_kl_weight` | 2.0 | вес KL в L_distill |
| `distill_l1_weight` | 5.0 | вес L1 в L_distill |
| `distill_giou_weight` | 2.0 | вес GIoU в L_distill |
| `distill_temperature` | 1.0 | T для Bernoulli KL (T² Hinton rescale) |
| `distill_min_weight` | 0.0 | флор для specific (только foreground) |
| `distill_general_min_weight` | 0.1 | флор для general (включая фон) |
| `distill_num_queries` | 300 | Q для specific branch |
| `distill_num_general_queries` | 100 | Q_g для general branch |
| `distill_general_query_std` | 0.02 | σ для random query_feat |

---

## 9. Потоки данных на GPU (один шаг)

```
DataLoader → (images [B,T,3,384,384],  targets,  teacher_centre [B,3,704,704])
                │                                        │
                ▼                                        ▼
         TemporalRFDETR                       FrozenRFDETRTeacher
     ┌───[backbone @ 384]──────┐              [backbone @ 704]
     │   [temporal_fusion]     │              [encoder]
     │   [encoder]             │              [two-stage block]
     │   [two-stage block]  ◄──┼──inject──   [decoder] ──► tgt(B,300,256)
     │   [decoder]             │   hook       refpoints(B,300,4)
     └──────────────────────── ┘              pred_logits(B,300,1)
                                              pred_boxes(B,300,4)
                                              foreground_weight(B,300)
```

Учитель прогоняется **два раза** (один для specific, один для general) под `torch.no_grad()` — без Autograd графа. Студент прогоняется **три раза** (Branch 1, 2, 3) все под `autocast`, loss суммируется, `backward()` один раз.
