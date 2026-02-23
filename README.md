# lerobot_dataset_v2_recorder_node


## Directory structure

```
dataset_root/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── <camera_name>/
│           ├── episode_000000.mp4
│           └── ...
└── meta/
    ├── info.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    └── stats.json
```
