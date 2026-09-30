# Experimental Results

We evaluate three cross-modal interaction configurations across three Transformer-based video backbones:

- **IBCA**: In-Block Cross-Attention
- **FCAF**: Final-stage Cross-Attention Fusion
- **FCCC**: Final Cross-Attention with Concatenation

Experiments are conducted on **First Impressions V2**, **KETI**, and **UDIVA V0.5**.

All results are reported as **mean ± standard deviation over five independent runs with different random seeds**.

- **First Impressions V2**: 1-MAE ↑
- **KETI**: 1-MAE ↑
- **UDIVA V0.5**: MSE ↓

The best-performing result for each backbone is shown in **bold**.

---

## First Impressions V2

**Metric: 1-MAE ↑ (higher is better)**

| Backbone | Video Only | IBCA | FCAF | FCCC |
|:---|---:|---:|---:|---:|
| TimeSformer | 0.9159 ± 0.0003 | 0.8989 ± 0.0009 | **0.9179 ± 0.0002** | 0.9173 ± 0.0005 |
| ViViT | 0.9172 ± 0.0002 | 0.9154 ± 0.0003 | **0.9188 ± 0.0004** | **0.9188 ± 0.0003** |
| VST | 0.9141 ± 0.0010 | 0.9059 ± 0.0079 | **0.9154 ± 0.0006** | 0.9151 ± 0.0004 |

---

## KETI

**Metric: 1-MAE ↑ (higher is better)**

| Backbone | Video Only | IBCA | FCAF | FCCC |
|:---|---:|---:|---:|---:|
| TimeSformer | 0.9120 ± 0.0007 | **0.9162 ± 0.0004** | 0.9107 ± 0.0028 | 0.9090 ± 0.0016 |
| ViViT | 0.9134 ± 0.0009 | 0.9094 ± 0.0011 | 0.9127 ± 0.0005 | **0.9136 ± 0.0009** |
| VST | 0.9086 ± 0.0011 | **0.9091 ± 0.0034** | 0.9086 ± 0.0019 | 0.9089 ± 0.0012 |

---

## UDIVA V0.5

**Metric: MSE ↓ (lower is better)**

| Backbone | Video Only | IBCA | FCAF | FCCC |
|:---|---:|---:|---:|---:|
| TimeSformer | 1.3085 ± 0.0403 | **1.1003 ± 0.0156** | 1.3198 ± 0.0498 | 1.3682 ± 0.0412 |
| ViViT | 1.2831 ± 0.0179 | 1.3626 ± 0.0159 | 1.3265 ± 0.0355 | **1.2794 ± 0.0142** |
| VST | 1.3306 ± 0.0150 | **1.2286 ± 0.0902** | 1.3054 ± 0.0291 | 1.3080 ± 0.0341 |

---
