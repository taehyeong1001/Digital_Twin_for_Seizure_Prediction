# survival_dist — survival rate 분포 예측·평가 코드

`final_code/`(SCDT-RC 원본)는 수정하지 않고 import만 한다. 문서는 `../docs/`에 있다.

| 파일 | 역할 |
|---|---|
| `survival.py` | 중도절단, Kaplan–Meier(Greenwood log-log 구간), Jeffreys 구간, 지수 탈출률 κ, x0별 집계 |
| `dist_metrics.py` | Brier / BSS / Murphy 분해 / reliability(consistency bar), BS(t) / IBS, D-calibration, logistic x50, CRPS, 불확실성 분해 |
| `selftest.py` | 위 두 모듈 검증 (scikit-survival 비교 + 합성 정답). 결과는 `outputs/selftest_report.txt` |
| `single_node_setup.py` | 원본 실행의 `protocol.json`에서 고정된 single node 설정을 재구성. `fit_member(seed)`로 seed만 바꾼 멤버 생성 |
| `plotting.py` | 공통 그림 스타일 (고정 색 순서 + 마커로 계열 구분) |
| `run_phase1_single_node.py` | 1단계: 기존 단일 RC를 시행 단위로 복원하고 새 지표로 채점 |
| `ensemble.py` | seed 앙상블 (멤버별 체크포인트, 건강 기록: non-finite, clip 비율, 첫 clip 시각). EESN `train_ensemble_checkpointed` 이식 |
| `run_phase2_single_node.py` | 2단계: 전이 구간 정밀 격자 + 20 멤버 앙상블, horizon 60/90/120/140 s별 분포 지표 |
| `run_multinode_pipeline.sh` | 다중 노드 원본 4단계 파이프라인을 `final_code/CODE` **사본**에서 실행 (결과가 사본 옆에 쌓임) |
| `run_multinode_ensemble.py` | 다중 노드 고정 모델(`--model rate|peak`)의 seed 앙상블 + 분포 보고서 |
| `run_arm_b_single_node.py` | 공정 비교 arm B (single node): 멤버 0 reservoir × innovation seed 20개 → `outputs/single_node/phase2/armB/` |
| `cross_table.py` | 공정 비교 교차표 (arm A/B/C/E/F × 기존 MAE, Brier, IBS, D-cal, cluster bootstrap 차이) → `outputs/cross_table/`. 해석은 `docs/04_fair_comparison.md` §6 |
| `critical_values.py` | seed별 생존 곡선 → seed별 임계값 λc 히스토그램 (arm B 잡음 분포, GT bootstrap과 비교) + 시간 보기 (S(t \| λ) heatmap, T50) → `outputs/critical_values/`. 해석은 `docs/05` §1b |
| `innovation_ablation.py` | docs/05 T1(잔차 ACF) + T2(innovation 방식만 바꾼 rollout: single iid/block/none, 다중 노드 block/iid_fullrate/none) + 분석 → `outputs/innovation_ablation/` |
| `dist_report.py` | 노드 수와 무관한 공용 분포 분석 (Brier/Murphy, x50(h), 불확실성 분해, IBS, D-cal, 이탈 종류, 그림 4장) |

## 실행 (저장소 루트에서)

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r final_code/requirements.txt scikit-survival

.venv/bin/python survival_dist/selftest.py

# 원본 single node 실행 (약 100 s). protocol.json을 만든다
cd final_code/CODE && ../../.venv/bin/python paper_single_node_scdt.py --out-dir ../../survival_dist/outputs/single_node/original && cd ../..

.venv/bin/python survival_dist/run_phase1_single_node.py   # 약 85 s
.venv/bin/python survival_dist/run_phase2_single_node.py --members 20 --workers 6   # 약 7분, 재실행하면 완료된 멤버는 건너뜀
.venv/bin/python survival_dist/run_phase2_single_node.py --members 20 --analyze-only   # 분석만

# 다중 노드 (3-node: 원본 사본 / 2-node: N_NODES=2로 패치한 사본, docs/03 참고)
mkdir -p survival_dist/outputs/three_node/run && cp -R final_code/CODE survival_dist/outputs/three_node/run/CODE
survival_dist/run_multinode_pipeline.sh survival_dist/outputs/three_node/run          # 약 40분
.venv/bin/python survival_dist/run_multinode_ensemble.py --run-dir survival_dist/outputs/three_node/run --label three_node --model rate --members 20 --workers 3
# 공정 비교 (docs/04): arm B 후 교차표
.venv/bin/python survival_dist/run_arm_b_single_node.py --rollouts 20 --workers 6   # 약 10분
.venv/bin/python survival_dist/run_multinode_ensemble.py --run-dir survival_dist/outputs/three_node/run --label three_node --model rate --arm-b 20 --workers 4
.venv/bin/python survival_dist/cross_table.py
.venv/bin/python survival_dist/run_multinode_ensemble.py --run-dir survival_dist/outputs/three_node/run --label three_node --model rate --cohort fine --arm-b 20 --workers 4   # 촘촘한 전이 cohort, 약 40분
.venv/bin/python survival_dist/critical_values.py   # 약 10 s
.venv/bin/python survival_dist/innovation_ablation.py acf --system three_node
.venv/bin/python survival_dist/innovation_ablation.py rollout --system three_node --members 5 --workers 2
.venv/bin/python survival_dist/innovation_ablation.py rollout --system single_node --members 5
.venv/bin/python survival_dist/innovation_ablation.py analyze
# BLAS 스레드 수는 기본값으로 둘 것 (바꾸면 멤버 0이 원본과 미세하게 달라진다)
```

출력: `outputs/single_node/phase1/` (`summary.json`, `per_x0.csv`, `trial_events.npz`, 그림 4개),
`outputs/single_node/phase2/` (`summary.json`, `per_x0.csv`, `per_member.csv`, `member_XXXX.npz`, `cohorts/`, 그림 4개),
`outputs/{two_node,three_node}/ensemble_report_{rate,peak}/` (`summary.json`, `per_level.csv`, `per_member.csv`, 그림 4개), 실행 사본과 멤버 파일은 `outputs/{two_node,three_node}/run/`.

공정 비교와 후속 분석 (docs/04, docs/05):
- arm B: `outputs/single_node/phase2/armB/`, `outputs/{two_node,three_node}/run/armB_rate[_fine]/`
- 촘촘한 전이 cohort: `outputs/{two_node,three_node}/run/fine_cohort.npz`, `ensemble_rate_fine/`. 교차 입력은 `outputs/{two_node,three_node}/cross_inputs_rate[_fine].npz`
- 교차표: `outputs/cross_table/` (`cross_table.md`, `.json`, `.csv`)
- seed별 임계값과 시간 보기: `outputs/critical_values/<system>[_fine]/` (`critical_value_hist[_logistic].png`, `survival_at_horizon_per_seed.png`, `survival_time_heatmap.png`, `t50_vs_lambda.png`, `per_seed.csv`, `summary.json`), 전체 요약은 `summary_all.json`
- T1/T2: `outputs/innovation_ablation/` (`acf/`, `t2_table.md`, `t2_summary.json`, `t2_*.png`, 방식별 멤버 파일)
