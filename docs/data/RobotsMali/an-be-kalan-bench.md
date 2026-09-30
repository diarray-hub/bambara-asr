---
language:
- bm
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
tags:
- audio
- speech
- low-resource-languages
- bambara
- education
dataset_info:
- config_name: duplicate
  features:
  - name: BookTitle
    dtype: string
  - name: sentenceID
    dtype: int64
  - name: text
    dtype: string
  - name: speakerAge
    dtype: int64
  - name: speakerGender
    dtype: string
  - name: speakerID
    dtype: string
  - name: duration
    dtype: float64
  - name: audio
    dtype: audio
  splits:
  - name: train
    num_bytes: 3749295635.885
    num_examples: 33481
  download_size: 4921102766
  dataset_size: 3749295635.885
- config_name: main
  default: true
  features:
  - name: BookTitle
    dtype: string
  - name: sentenceID
    dtype: int64
  - name: text
    dtype: string
  - name: speakerAge
    dtype: int64
  - name: speakerGender
    dtype: string
  - name: speakerID
    dtype: string
  - name: duration
    dtype: float64
  - name: audio
    dtype: audio
  splits:
  - name: train
    num_bytes: 184917975.603
    num_examples: 1203
  - name: test
    num_examples: 724
  download_size: 180050364
  dataset_size: 184917975.603
configs:
- config_name: duplicate
  data_files:
  - split: train
    path: duplicate/train-*
- config_name: main
  default: true
  data_files:
  - split: train
    path: main/train-*
  - split: test
    path: main/test-*
---
# Bambara Educational Speech Dataset

This dataset is a collection of READ Bambara text based on educational children's books from RobotsMali's GAIFE project. It is designed to support the training and benchmarking of Automatic Speech Recognition (ASR) models, with a particular focus on child speech, regional acoustics, and repetitive text structures (inherent to the domain).

The dataset is structured into two separate subsets to support specialized training and evaluation paradigms:
1. **`main`**: Contains separate training and test splits with disjoint recorded book titles and speaker IDs. Speaker IDs may not reliably identify distinct people across sessions.
2. **`duplicate`**: A highly dense, multi-speaker redundant training set featuring multiple recordings of the same source literature by a diverse pool of speakers.

---

## Dataset Architecture & Splits

The release has `main/train`, `main/test`, and `duplicate/train`. Exact `BookTitle` strings do not overlap between the test and either training split. The two training splits contain the same 22 title strings. Speaker IDs are metadata identifiers rather than verified distinct people; the separate `main` and `duplicate` training sets share IDs, while the test has no ID overlap with either training split at this revision. Dataset-derived totals below use the published `duration` metadata and count each utterance, including repeated readings.

### Summary Statistics

| Metric | `main/train` | `main/test` | `duplicate/train` | All published rows |
| :--- | :---: | :---: | :---: | :---: |
| Utterances | 1,203 | 724 | 33,481 | 35,408 |
| Total duration | 1.624 h | 0.889 h | 44.018 h | 46.531 h |
| Distinct speaker IDs | 8 | 11 | 113 | 124 |
| Distinct book titles | 22 | 17 | 22 | 39 |
| Mean audio duration | 4.86 s | 4.42 s | 4.73 s | 4.73 s |
| Audio duration range | 0.72–25.60 s | 0.32–16.48 s | 0.08–37.52 s | 0.08–37.52 s |
| Mean sentence length | 8.06 words | 7.18 words | 7.85 words | 7.84 words |
| Sentence length range | 1–26 words | 1–21 words | 1–26 words | 1–26 words |
| Rows missing any age, gender, or speaker ID | 0 | 38 | 0 | 38 |

The two training splits total **34,684 utterances and 45.642 h** (`main/train`: 1.624 h; `duplicate/train`: 44.018 h). All published rows, including the 0.889 h test split, sum to 46.531 h. These are retained dataset durations, not the 55 hours of raw campaign recordings.

### Demographic distributions

Known `speakerAge` values span **5–19** in this release. The test split has 2 utterances labelled age 5; these are metadata values that merit source-record verification. The test also has 38 rows with unknown age and gender. The age bands below count utterances, not unique children; 10–15 and 16–20 include both endpoints.

| Age band | `main/train` | `main/test` | `duplicate/train` | All published rows |
| :--- | :---: | :---: | :---: | :---: |
| Under 10 | 67 | 93 | 1,782 | 1,942 |
| 10–15 | 1,097 | 527 | 17,459 | 19,083 |
| 16–20 | 39 | 66 | 14,240 | 14,345 |
| Unknown | 0 | 38 | 0 | 38 |

| Gender metadata | `main/train` | `main/test` | `duplicate/train` | All published rows |
| :--- | :---: | :---: | :---: | :---: |
| Female | 783 | 522 | 16,688 | 17,993 |
| Male | 420 | 164 | 16,793 | 17,377 |
| Unknown | 0 | 38 | 0 | 38 |

At the current release revision, the test duration field places 692/724 utterances (95.6%) at or below 10 s and 719/724 (99.3%) at or below 15 s. These are metadata filter counts, not proof of a historical validation run's exact row count.

## Book inventory and age distribution

The table uses the exact published `BookTitle` strings. Ages are utterance counts in the order **under 10 / 10–15 / 16–20 / unknown**. `Speaker IDs` are distinct recorded identifiers within a title, not verified readers or complete-read counts.

### `main/train` — 22 titles

| BookTitle | Utterances | Duration (h) | Speaker IDs | Age counts (<10 / 10–15 / 16–20 / unknown) |
| :--- | ---: | ---: | ---: | :---: |
| Aminata Fari Fisayara | 68 | 0.089 | 2 | 67 / 1 / 0 / 0 |
| Bakɔrɔnin Saba | 45 | 0.057 | 1 | 0 / 45 / 0 / 0 |
| Bɛnkɛ Tɔm Ka So | 124 | 0.182 | 2 | 0 / 124 / 0 / 0 |
| Dawuda ni a Mɔkɛ | 38 | 0.054 | 1 | 0 / 38 / 0 / 0 |
| Dɔgɔtɔrɔ ni Farafinfurabɔla a | 97 | 0.144 | 1 | 0 / 97 / 0 / 0 |
| Filomani | 65 | 0.089 | 1 | 0 / 65 / 0 / 0 |
| Gawusu ni Masakɛ Sidiki | 51 | 0.071 | 1 | 0 / 51 / 0 / 0 |
| Gerenadi-Feerew | 47 | 0.081 | 1 | 0 / 47 / 0 / 0 |
| Gesedala Musa | 35 | 0.045 | 1 | 0 / 35 / 0 / 0 |
| Gundola Kuma | 59 | 0.064 | 2 | 0 / 59 / 0 / 0 |
| Kalo la Taama | 64 | 0.077 | 1 | 0 / 64 / 0 / 0 |
| Kan Orobotik | 39 | 0.046 | 1 | 0 / 39 / 0 / 0 |
| Korokara Yɛrɛdɔnbali | 74 | 0.099 | 2 | 0 / 74 / 0 / 0 |
| Kurun | 52 | 0.059 | 1 | 0 / 52 / 0 / 0 |
| Lamini Ka Don Kɛrɛnkɛrɛnnen | 77 | 0.099 | 1 | 0 / 77 / 0 / 0 |
| Mama ka Sama | 33 | 0.051 | 1 | 0 / 33 / 0 / 0 |
| Ne ni Mama ka Gafe Kalan | 38 | 0.037 | 1 | 0 / 38 / 0 / 0 |
| Saratu | 55 | 0.101 | 1 | 0 / 55 / 0 / 0 |
| Subahana Daga | 39 | 0.064 | 1 | 0 / 0 / 39 / 0 |
| Sɔminiminɛnw | 28 | 0.023 | 1 | 0 / 28 / 0 / 0 |
| Tulonkɛw | 51 | 0.057 | 2 | 0 / 51 / 0 / 0 |
| Yɛlɛ Ka Di Npogotiginin Mi Ye | 24 | 0.036 | 1 | 0 / 24 / 0 / 0 |

### `main/test` — 17 titles

| BookTitle | Utterances | Duration (h) | Speaker IDs | Age counts (<10 / 10–15 / 16–20 / unknown) |
| :--- | ---: | ---: | ---: | :---: |
| Anw Bɛ Baara Kɛ! | 55 | 0.106 | 3 | 50 / 5 / 0 / 0 |
| Ayisa ye nkalontigɛ dabila | 55 | 0.076 | 2 | 0 / 54 / 1 / 0 |
| Bako Cɛnin Ŋaniya Ɲuman | 46 | 0.079 | 1 | 0 / 46 / 0 / 0 |
| Bama Miirina | 24 | 0.024 | 2 | 22 / 0 / 1 / 1 |
| Cɛni Tulogɛlɛn | 46 | 0.057 | 1 | 0 / 9 / 0 / 37 |
| Denmisɛnya Kojuguw | 37 | 0.029 | 2 | 0 / 37 / 0 / 0 |
| Donfɛnw | 27 | 0.026 | 1 | 0 / 27 / 0 / 0 |
| Fali Nalonma Ni Ba Kegunma ani Ɲininkaliw ni u j | 39 | 0.057 | 2 | 0 / 39 / 0 / 0 |
| Jate | 36 | 0.039 | 1 | 0 / 0 / 36 / 0 |
| Ji Poyi Yɔrɔ | 72 | 0.058 | 1 | 0 / 72 / 0 / 0 |
| Kewale Numanw | 26 | 0.032 | 1 | 0 / 26 / 0 / 0 |
| Kogo | 39 | 0.075 | 1 | 0 / 39 / 0 / 0 |
| Kulɔriw | 30 | 0.033 | 1 | 0 / 30 / 0 / 0 |
| Mali kunkanko | 92 | 0.083 | 1 | 0 / 92 / 0 / 0 |
| Namasatigi | 21 | 0.027 | 1 | 21 / 0 / 0 / 0 |
| Ne ni Papa ka Gafe Kalan | 42 | 0.037 | 3 | 0 / 14 / 28 / 0 |
| Ni a tun bɛ se ... | 37 | 0.051 | 1 | 0 / 37 / 0 / 0 |

### `duplicate/train` — 22 titles

| BookTitle | Utterances | Duration (h) | Speaker IDs | Age counts (<10 / 10–15 / 16–20 / unknown) |
| :--- | ---: | ---: | ---: | :---: |
| Aminata Fari Fisayara | 2,420 | 2.792 | 38 | 172 / 1230 / 1018 / 0 |
| Bakɔrɔnin Saba | 1,596 | 2.131 | 38 | 126 / 790 / 680 / 0 |
| Bɛnkɛ Tɔm Ka So | 3,212 | 3.687 | 28 | 111 / 1503 / 1598 / 0 |
| Dawuda ni a Mɔkɛ | 1,179 | 1.585 | 31 | 40 / 590 / 549 / 0 |
| Dɔgɔtɔrɔ ni Farafinfurabɔla a | 2,106 | 2.868 | 24 | 194 / 850 / 1062 / 0 |
| Filomani | 2,124 | 2.552 | 32 | 134 / 1050 / 940 / 0 |
| Gawusu ni Masakɛ Sidiki | 1,473 | 2.022 | 30 | 148 / 856 / 469 / 0 |
| Gerenadi-Feerew | 1,330 | 2.100 | 29 | 48 / 753 / 529 / 0 |
| Gesedala Musa | 966 | 1.349 | 27 | 72 / 536 / 358 / 0 |
| Gundola Kuma | 1,710 | 2.257 | 28 | 177 / 941 / 592 / 0 |
| Kalo la Taama | 1,731 | 2.111 | 27 | 67 / 994 / 670 / 0 |
| Kan Orobotik | 1,200 | 1.833 | 30 | 83 / 581 / 536 / 0 |
| Korokara Yɛrɛdɔnbali | 1,873 | 2.357 | 25 | 0 / 1094 / 779 / 0 |
| Kurun | 1,314 | 1.859 | 27 | 0 / 736 / 578 / 0 |
| Lamini Ka Don Kɛrɛnkɛrɛnnen | 1,836 | 2.491 | 25 | 26 / 985 / 825 / 0 |
| Mama ka Sama | 826 | 1.415 | 25 | 0 / 442 / 384 / 0 |
| Ne ni Mama ka Gafe Kalan | 1,167 | 1.367 | 27 | 122 / 722 / 323 / 0 |
| Saratu | 1,360 | 2.353 | 26 | 58 / 672 / 630 / 0 |
| Subahana Daga | 1,198 | 1.580 | 29 | 84 / 652 / 462 / 0 |
| Sɔminiminɛnw | 1,236 | 1.160 | 32 | 120 / 564 / 552 / 0 |
| Tulonkɛw | 1,094 | 1.285 | 20 | 0 / 605 / 489 / 0 |
| Yɛlɛ Ka Di Npogotiginin Mi Ye | 530 | 0.867 | 20 | 0 / 313 / 217 / 0 |

Statistics above were computed from all metadata rows in the dataset's Parquet shards at revision [`95cf3103994ac13c13e086518c320a614cf75085`](https://huggingface.co/datasets/RobotsMali/an-be-kalan-bench/tree/95cf3103994ac13c13e086518c320a614cf75085). Sentence length counts whitespace-separated words; duration uses the published `duration` field. Audio was not decoded for this metadata audit.

---

## Critical Considerations: Features vs. Weaknesses

When developing models on this dataset, users should balance its unique profile against known recording constraints:

### 1. The High-Volume Duplication Matrix
* **As a Feature:** Due to the severe scarcity of open-source text and educational literature in Bambara, collecting deep audio variations on a finite set of text was a deliberate design choice. This split provides a robust playground for specific speech experiments, such as **acoustic multi-speaker verification, text-constrained acoustic profiling, and downstream speech representation probing**.
* **As a Weakness:** The textual diversity in the duplicate subset is inherently bottlenecked by the underlying literature. Models trained aggressively on the duplicate split without constraint can rapidly overfit to the vocabulary, tone structures, and phonetic bounds of these 22 recorded book titles.

### 2. Known Metadata Inconsistencies
* **The Identifier Inconsistency:** The release has 113 distinct speaker IDs across its training splits and 124 across all splits. On-the-ground project coordinators reported approximately 60 actual speakers; that estimate has not been verified from these metadata rows.
* **Implication:** This discrepancy highlights an operational metadata inflation error where individual speakers were assigned differing tracking IDs across different recording sessions, days, or environments. Users should exercise caution when benchmarking strict zero-shot speaker verification algorithms on this dataset without manual speaker clustering.

---

## Dataset Format

The Hugging Face release uses Parquet shards with `BookTitle`, `sentenceID`, `text`, `speakerAge`, `speakerGender`, `speakerID`, `duration`, and an `audio` feature. A local export can derive a NeMo-compatible JSON Lines (`.jsonl`) manifest. In that derived format, `audio_filepath` is a local file path, not a field of the published Parquet rows. For example:

```json
{
  "BookTitle": "Aminata Fari Fisayara",
  "sentenceID": 1,
  "text": "Mɔgɔ caman tun bɛ yen.",
  "speakerAge": 15,
  "speakerGender": "male",
  "speakerID": "spk_1778182779107_12bb3ea9c8",
  "duration": 2.56,
  "audio_filepath": "data/audios/dup_Aminata_Fari_Fisayara_1_1200.wav"
}
```

### Citation

If you utilize this dataset or its subsets in research, please cite the repository card details accordingly.

```
BIBTEX ENTRY COMING SOON
```
