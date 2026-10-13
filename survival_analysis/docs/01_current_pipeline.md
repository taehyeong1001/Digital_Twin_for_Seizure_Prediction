# 01. 기존 파이프라인 로직 기록 (`final_code/`)

> 목적: survival 분포 평가를 얹기 전에, **지금 코드가 실제로 무엇을 계산하는지**를 코드 기준으로 기록한다.
> README와 코드가 다른 곳은 코드 기준으로 적고 따로 표시했다.
> 작성 2026-10-12. single node는 코드를 직접 읽고 재실행해서 검증했다.
> 3-node는 상수와 이벤트 정의만 코드로 확인했고, 나머지는 README 기준이다 (5단계에서 다시 검증한다).

관련 문서: [02_metrics.md](02_metrics.md) (지표 정의), [03_progress_log.md](03_progress_log.md) (진행 기록과 결정), [../survival_distribution_plan.md](../survival_distribution_plan.md) (전체 아이디어와 문헌).

---

## 1. 공통 구성 요소

### 1.1 GT: Epileptor (`paper_epileptor_core.py`)

- Proix et al. (Brain, 2017)의 Epileptor 식(6변수: x1, y1, z, x2, y2, lowpass)을 쓴다.
  - 확률적 Heun 적분: dt = 0.04, 출력 256 Hz. 노이즈는 x2, y2에만 넣는다 (분산 0.0025).
- 관측 신호(proxy)는 **x2 − x1** (`source_proxy`)이다.
- 임계값 `X0_CRITICAL = −2.1`: 논문의 반올림값이다. 코드 주석에 따르면 축약식의 Hopf 점은 약 −2.062다.
- 느린 변수 z의 시간상수는 tau0 = 2857이다. 그래서 x0를 바꾼 뒤 발작으로 넘어가기까지 **수십 초**가 걸린다 (아래 결과 참고).

### 1.2 RC: 정적 조건부 sparse reservoir

- **single node:** `SparseStaticRC` (`paper_single_node_scrc.py`).
- **3-node:** `MultiOutputStaticRC` → `BlockInnovationRC` (`paper_multinode_coupled_scdt.py`, `experiment_three_node_scdt_block_innovation.py`).
- **상태 갱신:**
  `r ← (1−α) r + α tanh(A r + W_in u + W_static s)`
  - u: 동적 입력 (현재 신호값)
  - s: **static 입력** (통계 좌표, rollout 동안 고정)
  - α: leakage rate
  - A: spectral radius로 스케일한 sparse 행렬
- **readout:** `[r (홀수 인덱스는 제곱), u, 1]`에 ridge 회귀로 한 스텝 앞을 예측한다.
- **innovation 노이즈:** teacher forcing에서 얻은 residual(상하 0.25% 절단)을 rollout 매 스텝 무작위로 다시 뽑아 더한다 (× `innovation_scaling`).
  - 3-node의 block 모드는 연속된 residual 조각을 통째로 뽑아 짧은 시간 구조를 보존한다.
- **clip:** 예측값을 [min − 2.5·span, max + 1.2·span]으로 자른다.
  - 주의: `np.clip`은 NaN을 그대로 통과시킨다. 그래서 발산한 rollout은 이벤트 검출에서 **"생존"으로 집계될 수 있다**. 1단계 test cohort에서는 발산이 0건임을 확인했다. 앙상블 단계에서는 멤버별로 반드시 점검한다.
- **seed 하나가 전부 결정한다:** `RCParams.seed`가 W_in, W_static, A를 모두 정한다. → **seed 앙상블 = 이 seed만 바꾼 멤버들**.

### 1.3 rollout 프로토콜 ("switch")

1. GT를 **base x0**(정상 영역)에서 warmup 동안 시뮬레이션한 다음, **target x0**로 바꿔 미래 구간을 생성한다.
2. RC는 같은 GT warmup 신호로 warm-up한다. 이때 static = base x0의 통계 좌표다.
3. 그다음 static을 **target x0의 통계 좌표로 바꾸고** autonomous rollout을 한다.
4. **GT 시행과 RC 시행은 같은 warmup을 공유하는 짝(paired)이다.** 그래서 시행 단위로 비교할 수 있다.

### 1.4 survival rate는 무엇의 비율인가 — 시행 사이에 무엇이 다른가 (2026-10-13 추가)

**결론:** final_code의 RC survival rate는 **reservoir 하나로 고정한 채, 초기 상태와 노이즈 경로만 바꿔 반복한 rollout에서 이탈하지 않은 비율**이다.

- **시간 곡선이 아니다.** x0 수준마다 "horizon H까지 이벤트가 없었던 시행의 비율", 즉 S(H) 값 하나를 계산하고, 이를 x0(3-node는 max x0)에 대해 그린다.
- **이벤트 시각은 계산하지만 쓰지 않는다.** `grouped_survival`은 이벤트가 있었는지만 쓴다.
- **비교 지표는 하나뿐이다.** `survival_mae = mean_x0 |S_GT(H) − S_RC(H)|`. x0 수준마다 같은 가중치를 준다.

**같은 x0 수준 안에서, 시행 사이에 무엇이 같고 무엇이 다른가:**

| 요소 | GT 시행들 | RC 시행들 | 코드 |
|---|---|---|---|
| 모델 | 같은 Epileptor 식 | **같은 reservoir 하나** (W_in, W_static, A, W_out 고정) | `rc.fit` 1회 |
| 조건 (x0 / static) | 같은 target x0 | 같은 target static 좌표 | `static_from_x0`, `coordinate_model.predict` |
| 초기 상태 | 다르다 (warmup의 GT 노이즈 seed가 다름) | 다르다 (각자 짝지어진 GT warmup으로 warm-up하므로 rollout 시작 시 reservoir 상태와 마지막 입력이 다름) | `build_switch_cohort`, `simulate_switch_cohort` |
| 미래의 노이즈 경로 | 다르다 (확률적 Heun) | 다르다 (innovation 노이즈: RNG seed는 batch 전체에 하나지만 **batch의 열(시행)마다 따로 뽑는다**) | `paper_single_node_scrc.py:248`, `experiment_three_node_scdt_block_innovation.py:106~` |
| reservoir seed | — | **바뀌지 않는다** | — |

- 그래서 GT와 RC 비율은 둘 다 "주어진 x0에서 초기 상태와 노이즈가 무작위일 때 H 안에 이탈할 확률"을 같은 방식으로 추정한 값이다 (각각 8–10개 표본).
- **final_code에 빠져 있는 변동원은 reservoir 자체의 무작위성(reservoir seed)이다.** 즉 RC 곡선은 "그 reservoir 하나에 대한 조건부 추정"이다. 우리 seed 앙상블은 이 축을 추가한다 ([04](04_fair_comparison.md)).
- 우리 불확실성 분해의 용어로 바꾸면: final_code의 RC 비율 = 멤버 하나의 p_j다. 이 p_j가 담고 있는 시행 간 변동이 aleatoric 항이다 ([02](02_metrics.md)의 `uncertainty_decomposition`).

**시스템별 수치 (코드 기준):**

| | single node | 3-node |
|---|---|---|
| 이벤트 | 첫 국소 최소 ≤ −2.0, blanking 없음 | 어느 노드든 한 샘플이라도 ≤ −0.5, 처음 5 s 무시 |
| warmup (base x0) | 20 s (−2.18) | 40 s (−2.30, 모든 노드). warmup 중 이벤트가 나면 seed를 교체하지 않고 에러로 중단 |
| H | 140 s | 120 s |
| test 격자 | x0 26개 × seed 10 = 260 | max x0 33단계(−2.50 ~ −1.90) × seed 8 = 264 (`block_peak_bo100.TEST_LEVELS`). 각 프로필은 노드 하나가 max, 나머지는 max − U(0.025, 0.10) |
| 비율 해상도 | 0.1 | 0.125 |
| RC rollout | 시행당 1회, innovation seed `args.seed + 90000` | 시행당 1회, innovation seed `SEED + 3200` (peak 단계) |
| 그림의 50% 지점 | — | GT 곡선이 0.5를 지나는 인접 두 점의 선형 보간 (사후값, 문헌의 −2.1과 다름) |

---

## 2. Single node (`paper_single_node_scdt.py`) — 코드로 검증함

| 항목 | 값 |
|---|---|
| 학습 x0 | −2.30, −2.27, −2.24, −2.21, −2.18, −2.15 (전부 임계 전, 각 160 s 정상 궤적 1개) |
| static 통계 | 26개 후보에서 x0와의 상관과 중복 제거로 2개 선택. 이번 실행에서는 **permutation_entropy, std** |
| 통계 좌표 | 학습 x0에 선형 적합한 `static_from_x0` (x0 → 통계). 즉 test에서는 **x0로 계산한 통계**를 넣는다 |
| base x0 / warmup / rollout | −2.18 / 20 s / **140 s** (horizon) |
| 샘플링 | 256 Hz를 8배 간축해 32 Hz |
| 하이퍼파라미터 | GP-BO 24회, reservoir 280 노드. 검증 cohort는 x0 8개 × 4 seed |
| test cohort | x0 26개 (−2.2925 … −1.9275, 간격 0.0146) × **10 seed** = 260 시행 |
| 재현 결과 | BO 선택 seed 160851, test survival MAE 0.0769. 저장소의 `final_figures` 그림과 동일함을 확인 |

### 2.1 이벤트(정상 이탈) 정의 — README와 다름

- **코드:** `detect_local_minimum_onset`. proxy가 **−2.0 이하인 첫 국소 최소**의 시각이다.
  - 측정 기준은 rollout(= target 전환) 시작부터의 초다.
  - blanking 없음. 전환 직후 0초부터 검출한다.
- README에 적힌 "5초 이후 −0.5 이하"는 **3-node의 정의**다. single node와는 다르다.
- GT와 RC 모두 **같은 함수**로 검출한다. (참고: 별도 함수 `survival_scan`은 RC 쪽에 feature-manifold 이탈을 쓰지만, 메인 실험은 이 함수를 쓰지 않는다.)

### 2.2 survival 계산

- `grouped_survival`: x0마다 `mean(이벤트 없음)`을 계산한다. 이벤트는 rollout 140 s 안에서만 검출되므로 **horizon = 140 s 시점의 생존 비율**이 된다.
- RC는 GT 시행마다 rollout을 1회 한다. 즉 x0마다 RC 10회, GT 10회다. innovation RNG seed는 batch 전체에 하나지만 시행마다 다른 노이즈를 뽑는다 (§1.4).

### 2.3 BO 목적함수 (참고)

`score = 6·false_collapse + 12·missed_collapse + 1.5·monotonic + 0.25·NRMSE(10 s) + 0.35·|log std 비|`

- 여기에 **이탈 시점 정보는 없다.** horizon 시점의 비율과 정상 구간의 파형만 본다.
- 1단계에서 RC가 이탈 *여부*는 비슷하게 맞추지만 *시점*은 크게 틀리는 것으로 나왔다. 그 이유가 이 목적함수 구성일 수 있다.

---

## 3. 3-node (README 기준, 일부 코드 확인)

| 항목 | 값 |
|---|---|
| GT | 3-node 결합 Epileptor, x0,1, x0,2, x0,3를 **모두** 바꾼다. base x0 = −2.30 |
| warmup / horizon | 40 s / **120 s** (코드 확인) |
| 이벤트 | `first_threshold_crossing`: 어느 노드든 proxy ≤ **−0.5**. **처음 5 s는 무시**(blanking) (코드 확인) |
| static | 노드별 mean / std / lag-1 AC, 총 9개. x0(3차원) → 통계는 ridge 회귀 |
| RC | 1,500 노드, block innovation, BO 100회 + peak 가중 BO 100회 + 독립 cohort 확인 |
| x축 | max(x0,1, x0,2, x0,3) |
| 주의 | README에 따르면 일부 `event_signals`(이탈 궤적)를 readout 학습과 residual에 쓴다. 실제 EEG로 옮길 때는 정보 누출 위험이 있다 |

- **blanking 5 s에 대한 메모:** single node에서 RC가 전환 직후 1 s 안에 이탈한 현상(1단계 결과)이 3-node에서도 있다면, blanking이 그것을 가리고 있을 수 있다. 5단계에서 확인한다.

---

## 4. 1단계에서 새로 드러난 사실 (2026-10-12)

자세한 수치는 [03_progress_log.md](03_progress_log.md)에 있다.

1. **GT의 140 s 생존은 이 격자에서 사실상 결정론적이다.**
   - x0 ≤ −2.0735이면 10/10 생존, x0 ≥ −2.0589이면 0/10 생존이다.
   - 독립 seed로 다시 만든 cohort도 26개 x0 모두에서 같은 결과가 나왔다 (oracle Brier = 0).
   - 그래서 "horizon 시점의 비율"만 보면 GT 쪽 확률성이 보이지 않는다. 확률성은 **0.0146보다 좁은 전이 구간**과 **이탈 시점**에만 있다.
2. **GT 이탈 시점은 넓게 퍼져 있다.**
   - x0마다 표준편차가 약 5–10 s이고, 중앙값은 83 s(x0 = −2.059)에서 36 s(x0 = −1.93)로 짧아진다.
   - 확률적 정보는 **시간 축**에 있다.
3. **RC는 이탈 여부는 비슷하게 맞추지만 시점이 크게 틀린다.**
   - x0 ≥ −2.0005이면 RC는 전환 후 **1 s 안에** 이탈한다. GT는 40–50 s 걸린다.
   - static 전환에 즉시 반응하는 RC와 느린 변수 z(tau0 = 2857)를 따라가는 GT의 시간 척도가 다르다.
4. RC는 GT가 한 번도 이탈하지 않는 x0 = −2.1611(임계 전)에서 거짓 이탈 2건을 냈다.

## 5. 2단계에서 새로 드러난 사실 (2026-10-12)

5. **GT의 생존은 결정론적 분기로 정해진다.**
   - 정밀 탐색 결과 (seed 20개/x0): x0 ≤ −2.06135는 140 s 동안 한 번도 이탈하지 않고, x0 ≥ −2.06130은 전부 이탈한다 (이탈 시점 80–116 s).
   - 전이 폭은 0.00005 이하이고, 위치는 코드 주석의 축약식 Hopf 점(약 −2.062)과 거의 같다.
   - 분기점 바로 위에서도 이탈 시점은 무한대로 늘어나지 않고 약 100 s에 머문다. 이탈까지의 시간이 x0까지의 거리보다 느린 변수 z의 이완(tau0)에 주로 좌우되기 때문으로 보인다.
   - 이 노이즈 수준(분산 0.0025)에서는 **노이즈로 인한 확률적 생존 구간이 사실상 없다.** 확률성은 이탈 시점에만 있다.
6. **RC의 "이탈"은 전부 clip 포화다** (멤버 0 기준).
   - x0 ≥ −2.03에서 RC 궤적은 rollout 시간의 87–99%를 clip 경계에 붙어 있다. 임계 전 x0 = −2.1611의 거짓 이탈도 clip 비율이 최대 0.67이다.
   - 즉 RC는 static 입력이 학습 범위(x0 ≤ −2.15)를 크게 벗어나면 발산한다. 그러면 clip이 잡아 둔 일정한 값이 "−2.0 이하의 국소 최소"로 검출된다.
   - 그래서 생존 곡선이 GT와 비슷해 보이는 것은 "RC가 발산하기 시작하는 x0"가 GT 분기점 근처에 있기 때문이다. 발작 같은 동역학을 재현해서가 아니다.
   - 범위 메모: 모델은 고정하고 평가만 하기로 했다 (03 로그). 원래 이벤트 정의로 지표를 계산하되, 이탈 종류(포화 vs 동역학)를 함께 보고한다 ([02](02_metrics.md) §6.4).

## 6. 다중 노드 재실행에서 확인한 것 (2026-10-13)

- 3-node 4단계 파이프라인을 원본 코드 사본에서 다시 실행했다. 1–3단계는 정상이었지만 **4단계(확인)에서 후보가 선택되지 않아 실패했다** (자세한 내용은 [03](03_progress_log.md)). 저장소의 robust 최종 모델은 재현되지 않았다.
- 다중 노드 BO는 모든 시행에서 reservoir seed를 `SEED + 900`으로 고정한다. single node BO는 시행마다 seed가 달랐다.
- 다중 노드 BO의 선택 조건에는 "clip 비율 0"이 있다. 실제로 재실행한 rate / peak 모델의 seed 앙상블에서 clip 포화 이탈은 0건이었다.
- 3-node 학습 데이터 = 정상 궤적 + 이탈 궤적(`event_signals`). readout과 residual 모두 이탈 동역학을 본다.
- 2-node 변형(`N_NODES = 2`, 결합 [[0,1],[1,0]])은 [03](03_progress_log.md)에 설계와 패치 목록이 있다.
- rollout의 부동소수점 결과는 BLAS 스레드 수에 따라 달라진다. 이탈 여부는 같지만, 이탈 시각이 최대 1.8 s 달라질 수 있다.
