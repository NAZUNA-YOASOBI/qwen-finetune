# VRSBench Results

## 13 类结果

| shot | 变体 | mean IoU | acc@0.5 | acc@0.7 | valid_for_iou | 指标文件 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1-shot | `v1` | `0.4927` | `0.5502` | `0.3459` | `16077` | `vg_eval_metrics_1shot_v1.json` |
| 1-shot | `v2` | `0.4931` | `0.5526` | `0.3464` | `16085` | `vg_eval_metrics_1shot_v2.json` |
| 1-shot | `v3` | `0.4699` | `0.5177` | `0.3155` | `16097` | `vg_eval_metrics_1shot_v3.json` |
| 1-shot | `v4` | `0.4187` | `0.4446` | `0.2447` | `16110` | `vg_eval_metrics_1shot_v4.json` |
| 5-shot | `v1` | `0.5092` | `0.5758` | `0.3643` | `16021` | `vg_eval_metrics_5shot_v1.json` |
| 5-shot | `v2` | `0.5163` | `0.5845` | `0.3758` | `16042` | `vg_eval_metrics_5shot_v2.json` |
| 5-shot | `v3` | `0.5139` | `0.5799` | `0.3712` | `16026` | `vg_eval_metrics_5shot_v3.json` |
| 5-shot | `v4` | `0.5180` | `0.5848` | `0.3762` | `16012` | `vg_eval_metrics_5shot_v4.json` |
| 10-shot | `v1` | `0.5255` | `0.6025` | `0.3941` | `16137` | `vg_eval_metrics_10shot_v1.json` |
| 10-shot | `v2` | `0.5300` | `0.6072` | `0.3995` | `16135` | `vg_eval_metrics_10shot_v2.json` |
| 10-shot | `v3` | `0.5349` | `0.6155` | `0.4107` | `16143` | `vg_eval_metrics_10shot_v3.json` |
| 10-shot | `v4` | `0.5326` | `0.6115` | `0.4091` | `16142` | `vg_eval_metrics_10shot_v4.json` |

## 26 类结果

| shot | 变体 | mean IoU | acc@0.5 | acc@0.7 | valid_for_iou | 指标文件 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1-shot | `v1` | `0.3993` | `0.3699` | `0.2268` | `13793` | `vg_eval_metrics_fullclip_1shot_v1.json` |
| 1-shot | `v2` | `0.4070` | `0.3814` | `0.2323` | `13863` | `vg_eval_metrics_fullclip_1shot_v2.json` |
| 1-shot | `v3` | `0.3788` | `0.3368` | `0.1934` | `13596` | `vg_eval_metrics_fullclip_1shot_v3.json` |
| 1-shot | `v4` | `0.3552` | `0.3055` | `0.1643` | `13536` | `vg_eval_metrics_fullclip_1shot_v4.json` |
| 5-shot | `v1` | `0.5223` | `0.5924` | `0.3973` | `16134` | `vg_eval_metrics_fullclip_5shot_v1.json` |
| 5-shot | `v2` | `0.5282` | `0.6015` | `0.4030` | `16138` | `vg_eval_metrics_fullclip_5shot_v2.json` |
| 5-shot | `v3` | `0.5301` | `0.6037` | `0.4072` | `16129` | `vg_eval_metrics_fullclip_5shot_v3.json` |
| 5-shot | `v4` | `0.5314` | `0.6063` | `0.4070` | `16134` | `vg_eval_metrics_fullclip_5shot_v4.json` |
| 10-shot | `v1` | `0.5353` | `0.6088` | `0.4133` | `16131` | `vg_eval_metrics_fullclip_10shot_v1.json` |
| 10-shot | `v2` | `0.5371` | `0.6117` | `0.4144` | `16134` | `vg_eval_metrics_fullclip_10shot_v2.json` |
| 10-shot | `v3` | `0.5461` | `0.6252` | `0.4313` | `16142` | `vg_eval_metrics_fullclip_10shot_v3.json` |
| 10-shot | `v4` | `0.5484` | `0.6259` | `0.4363` | `16108` | `vg_eval_metrics_fullclip_10shot_v4.json` |

## v1-v4 含义

- `v1`: language
- `v2`: language + merger
- `v3`: language + vision_mlp + merger
- `v4`: language + vision_attn + vision_mlp + merger
