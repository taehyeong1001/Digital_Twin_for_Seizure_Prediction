# 02. 지표 정의서 — 무엇을, 왜, 어떻게 계산하나

> 코드: `survival_dist/survival.py`, `survival_dist/dist_metrics.py`
> 검증: `survival_dist/selftest.py` (18개 항목, scikit-survival과 수치 비교 포함)
> 문헌 근거와 확인 수준: [../survival_distribution_plan.md](../survival_distribution_plan.md) §2

각 지표마다 다음 순서로 적는다: **정의 → 식 → 읽는 법 → 왜 쓰나 → 구현 메모 → 근거**.

---

## 0. 기본 대상과 표기

| 기호 | 의미 |
|---|---|
| x0 | 목표 파라미터 (단일 노드의 excitability. 3-node는 그림의 x축으로 max(x0)) |
| T | **정상 이탈 시점**. rollout 시작(target 전환)부터의 초. 이벤트 정의는 [01](01_current_pipeline.md) §2.1, §3 |
| H | horizon. single node는 140 s, 3-node는 120 s |
| o_i | 시행 i의 결과. 1 = H 안에 이탈, 0 = 생존 |
| p | 이탈 **확률** 예측값. 생존 확률은 1 − p |
| S(t) | 생존함수 P(T > t) |
| 멤버 j | seed만 다른 reservoir (2단계부터) |

**중도절단 (censoring).** H까지 이탈하지 않은 시행은 "T > H"라는 것만 안다. 이를 우측 중도절단이라고 한다.
- 지금 데이터는 모든 시행이 H라는 같은 시점에서만 잘린다 (administrative censoring). 그래서 대부분의 보정이 단순해진다.
- 코드: `survival.censor(event_times, H)`는 `(durations, observed)`를 돌려준다. H와 정확히 같은 시각의 이벤트는 관측된 것으로 센다 (원본 코드와 같은 규칙).

---

## 1. 비율과 그 불확실성

### 1.1 horizon 생존율 `survival_at_horizon`
- **정의:** 시행 중 [0, H]에서 이탈하지 않은 비율 = 원본 코드의 `grouped_survival`.
- **검증:** Kaplan–Meier S(H)와 정확히 같아야 한다. 1단계 통과 조건이었고 통과했다.

### 1.2 Jeffreys 구간 `jeffreys_interval`
- **정의:** k번 이탈 / n번 시행일 때 Jeffreys 사전분포의 사후분포 **Beta(k + ½, n − k + ½)** 에서 2.5% / 97.5% 분위를 구한다.
  - 관례상 k = 0이면 하한 0, k = n이면 상한 1로 둔다.
- **읽는 법:** "이 시행 수로는 진짜 비율이 이 범위 어디든 가능하다."
  - n = 10에서 10/10 생존이어도 95% 구간은 대략 [0.78, 1]이다.
- **왜 쓰나:** 흔히 쓰는 Wald 구간 √(p(1−p)/n)은 p가 0이나 1이면 폭이 0이 되고, 작은 n에서 실제 coverage가 크게 모자란다. Brown, Cai & DasGupta (2001)는 Wilson이나 Jeffreys를 권한다.
  - Jeffreys는 Bayesian 사후분포와 같다. 그래서 "GT 비율의 분포"를 표본으로 뽑을 때도 그대로 쓸 수 있다 (`jeffreys_posterior_samples`).
- **selftest:** p = 0.3, n = 40에서 coverage 0.944 (명목 0.95).

---

## 2. 이탈 시점의 분포 (생존분석)

### 2.1 Kaplan–Meier 곡선 `kaplan_meier`
- **정의:** 이벤트 시각 t_k마다 S(t) = Π_{t_k ≤ t} (1 − d_k / n_k).
  - d_k: 그 시각의 이벤트 수
  - n_k: 그 시각 직전까지 남아 있는 시행 수 (at risk)
- **구간:** Greenwood 분산 Var S = S² Σ d_k / (n_k (n_k − d_k)). 이를 **log(−log) 변환**(Kalbfleisch–Prentice)으로 바꿔 [0, 1] 안에 머무르는 95% 구간을 만든다.
- **읽는 법:** 곡선이 빨리 떨어질수록 이탈이 빠르다. 같은 x0에서 GT와 RC 곡선을 겹쳐 보면 시점 분포가 맞는지 바로 보인다.
- **구현 메모:** EESN `R0c_metrics_and_plots.jl`의 `kaplan_meier`를 옮기고 Greenwood 구간을 더했다.
- **selftest:** scikit-survival `kaplan_meier_estimator(conf_type="log-log")`와 S, 하한, 상한이 모두 1e-16 수준으로 일치한다.

### 2.2 반감기 `KMCurve.half_life`
- S(t) ≤ 0.5가 되는 첫 t (EESN `km_half_life`). 중앙 이탈 시점이다.

### 2.3 지수 탈출률 κ `exponential_escape_rate`
- **정의:** 위험률(hazard)이 일정하다고 가정한 S(t) = e^{−κt}의 최대우도 추정. **κ̂ = (이벤트 수) / (모든 시행의 관측 시간 합)**. 구간은 χ² 피벗으로 구한다.
- **읽는 법:** 1/κ는 평균 탈출 시간(MFPT)이다. x0에 따른 κ(x0) 곡선은 "임계에서 멀어질수록 얼마나 빨리 무너지나"를 한 숫자로 요약한다.
- **왜 쓰나:** 노이즈로 인한 탈출(Kramers형)과 transient chaos에서는 생존확률이 대체로 지수적으로 감소한다 (plan §2.2b).
- **주의:** GT Epileptor는 느린 변수 z가 결정론적으로 흘러가서 이탈한다. 그래서 처음에는 위험률이 0에 가깝다가 나중에 급히 커지는 **지연된 분포**에 가깝다 (1단계 KM 곡선 참고). 지수 가정이 맞지 않으므로 κ는 보조 요약값으로만 쓰고, 주 비교는 KM 곡선과 IBS로 한다.
- **selftest:** 진짜 κ = 0.015, n = 4000 → 추정 0.0154, 구간 [0.0149, 0.0159].

---

## 3. 이진 사건 확률의 점수 (Brier 계열) — **주 지표**

### 3.1 Brier score `brier_score`
- **정의:** BS = (1/N) Σ (p_i − o_i)². 예측 확률과 실제 0/1 결과의 제곱 오차다.
- **읽는 법:**
  - 0이면 완벽하다.
  - 항상 50%라고 말하면 0.25다.
  - 낮을수록 좋다.
- **왜 주 지표인가:**
  1. **proper scoring rule**이다. 자신이 믿는 확률을 그대로 말할 때 기대 점수가 가장 좋다. 그래서 확률을 일부러 극단으로 몰거나 흐리게 해서 점수를 올릴 수 없다 (Gneiting & Raftery 2007).
  2. **시행 단위**로 계산한다. RC가 낸 확률 p(x0)를 GT의 각 시행 결과 o_i와 맞춰 본다. 그래서 GT 쪽도 유한 표본이라는 문제가 자연스럽게 처리된다.
     - 반대로 "x0별 GT 비율 대 RC 비율"을 비교하는 방식(지금의 survival MAE)은 GT 비율 자체의 잡음을 무시한다.
  3. seizure forecasting 문헌(Proix et al. 2021 등)이 Brier skill score로 보고하므로 비교하기 좋다.
- **1단계에서 RC 예측 확률:** p_RC(x0) = (그 x0에서 RC가 이탈한 rollout 수) / 10. 2단계부터는 앙상블 평균 p̄(x0)를 쓴다.

### 3.2 Brier skill score `brier_skill_score`
- **정의:** BSS = 1 − BS(모델) / BS(기준).
  - 기준은 보통 **climatology**: 모든 시행에 전체 평균 이탈률을 똑같이 말하는 예측 (`climatology`).
- **읽는 법:** 1이면 완벽, 0이면 기준과 같고, 음수면 기준보다 나쁘다.

### 3.3 Murphy 분해 `murphy_decomposition`
- **정의:** BS = **REL − RES + UNC**.
  - REL (reliability, 낮을수록 좋음) = Σ_k n_k (p̄_k − ō_k)² / N. "70%라고 말한 경우들에서 실제로 70%가 일어났나"를 본다. 보정이 나쁜 정도다.
  - RES (resolution, 높을수록 좋음) = Σ_k n_k (ō_k − ō)² / N. 예측이 상황을 얼마나 **구분**하나를 본다.
  - UNC (uncertainty) = ō(1 − ō). 데이터 자체의 불확실성이고 모델과 무관하다.
- **구현 메모:** 예측값이 몇 개로만 나오면(지금처럼 0, 0.1, …, 1) 서로 다른 값끼리 묶는다. 이때 식이 **정확히** 성립한다 (selftest residual ≈ 1e-17). 연속적인 예측이면 bin으로 묶고 남는 오차를 `residual`로 돌려준다.

### 3.4 Reliability diagram + consistency bar `reliability_curve`
- **정의:** 예측 확률을 10개 bin으로 나누고, bin마다 (평균 예측, 실제 이탈 빈도)를 찍는다. 대각선에 있으면 보정된 것이다.
- **consistency bar (Bröcker & Smith 2007):** "이 예측이 **완벽히 보정되어 있다고 가정**하면, 같은 수의 시행에서 관측 빈도가 우연히 어디까지 흔들릴 수 있나"를 보여 준다.
  - 만드는 법: 예측값을 복원추출하고, o* ~ Bernoulli(p*)로 결과를 다시 뽑아 bin별 빈도의 90% 범위를 구한다.
  - 점이 막대 밖에 있으면 표본 잡음으로는 설명되지 않는 **보정 오류**다.
- **주의:** 표본이 적은 bin은 막대가 길다. 0이나 1 bin은 막대 길이가 0이다 (p = 0이면 재표본도 항상 0이므로).

---

## 4. 시간에 따른 생존 예측의 점수

### 4.1 시간별 Brier score와 integrated Brier score (IBS) `brier_score_over_time`, `integrated_brier_score`
- **정의 (Graf et al. 1999):** 각 시각 t에서 "T > t인가"를 예측한 S_i(t)를 Brier로 채점한다. 중도절단은 검열 분포의 KM G로 역가중한다.
  BS(t) = (1/N) Σ [ S_i(t)² · 1(T_i ≤ t, 이탈) / G(T_i) + (1 − S_i(t))² · 1(T_i > t) / G(t) ]
- **IBS** = BS(t)를 시간 격자에서 사다리꼴로 적분하고 구간 길이로 나눈 값이다 (scikit-survival과 같은 규약).
- **읽는 법:** 0이 완벽하다. 이탈 **여부**와 **시점**을 함께 채점한다. horizon 생존율이 같아도 시점 분포가 틀리면 IBS가 나빠진다.
- **기준선:** x0를 무시하고 모든 시행을 합친 KM 곡선 하나로 예측한 IBS. 생존분석에서 공변량이 없는 표준 null 모델이다.
- **구현 메모:** 지금은 모든 검열이 H에서만 일어나므로 [0, H) 구간에서 G = 1이고 가중치가 사라진다. 시간 격자는 H보다 짧게 잡는다 (1–139 s).
- **selftest:** 무작위 검열을 섞은 데이터에서 scikit-survival의 `brier_score`, `integrated_brier_score`와 1e-10 수준으로 일치한다.

### 4.2 D-calibration `d_calibration`
- **정의 (Haider et al. 2020):** 예측 생존곡선이 맞다면, 각 시행의 **u_i = S_i(T_i)** 는 Uniform(0, 1)을 따라야 한다. 생존분석판 PIT다.
  - 이탈한 시행: u_i를 10개 bin 중 하나에 넣는다.
    - 예측이 계단 함수라 점프 위에 질량이 몰리므로, [S(T), S(T−)] 사이에서 균등하게 무작위로 뽑는다 (randomized PIT).
  - 중도절단된 시행 (c = S_i(H)): 진짜 u는 c보다 작다는 것만 안다. 그래서 질량 1을 [0, c)에 균등하게 나눈다.
  - bin 질량이 균등한지 χ² 검정으로 본다 (자유도 9).
- **읽는 법:**
  - p-value가 작으면 시점 분포가 체계적으로 틀렸다는 뜻이다.
  - 질량이 **낮은 u 쪽**에 몰리면, 실제 이탈이 예측보다 **늦다**는 뜻이다 (예측 S가 이미 작아진 뒤에 이탈). 반대 방향이면 실제 이탈이 예측보다 빠르다.
- **selftest:** 진짜 모델은 p = 0.14로 통과하고, 위험률을 2배로 잡은 모델은 p ≈ 0으로 기각한다.

---

## 5. x0에 따른 곡선 요약 (dose–response)

### 5.1 logistic 적합 `fit_logistic_survival`
- **정의:** 생존 S(x0) = 1 / (1 + exp((x0 − x50)/s))를 binomial 최우도로 적합한다.
  - **x50:** 생존 50% 지점. 독성학의 LD50에 해당한다.
  - **s:** 전이 폭. 작을수록 날카롭다.
- **왜 쓰나:** 곡선 전체 대신 "임계점 위치"와 "전이의 날카로움"이라는 물리적으로 해석되는 두 숫자로 GT와 RC를 비교한다.
- **주의 (완전 분리):** 모든 x0에서 결과가 0/10 또는 10/10이면(1단계 GT가 이렇다), 최우도 추정의 s가 0으로 수렴한다. 그러면 x50은 마지막 생존 x0와 첫 이탈 x0 **사이 어디든** 될 수 있다.
  - 이때 bootstrap 구간 폭 0은 "확실하다"는 뜻이 아니라 **해상도 부족**이라는 뜻이다. `n_mixed_x0`(결과가 섞인 x0 수)를 같이 보고한다.
  - 해법: 전이 구간 안의 x0 격자를 촘촘히 한다 (Fieller와 delta 구간도 x50이 실험 범위에서 멀거나 사이에 끼면 불안정하다).
- **selftest:** x50 = −2.08, s = 0.02, 시행 200개/x0에서 (−2.0786, 0.0201)을 복원했다.

### 5.2 bootstrap `bootstrap_logistic`
- x0마다 시행을 복원추출(= k* ~ Binomial(n, k/n))하고 다시 적합해 (x50, s)의 분포를 얻는다.
- 2단계에서는 **멤버별 적합**으로 얻는 x50 분포(모델 불확실성)와 이 bootstrap 분포(GT 표본 불확실성)를 겹쳐 그린다.

### 5.3 horizon별 x50(h) (2단계 추가)
- **배경:** single node GT에서는 H = 140 s 생존율이 x0에 대해 **폭 0.00005 이하의 계단**이다 (03 로그, 2단계).
  - 분기점 x0* ≈ −2.0613 위에서는 모든 시행이 80–116 s 사이에 이탈하고, 아래에서는 한 번도 이탈하지 않는다.
  - 그래서 하나의 H만 보면 GT 쪽 분포가 없다.
- **정의:** horizon h ∈ {60, 90, 120, 140 s}마다 "h초 안 이탈"을 사건으로 두고 5.1의 logistic을 적합한다. x50(h)는 "h초 안에 절반이 이탈하는 x0"다.
  - h가 짧으면 이탈 시점의 퍼짐 때문에 넓은 x0 범위에서 0 < S < 1이 된다. 그러면 x50(h)와 s(h)가 의미 있는 분포 비교 대상이 된다.
- **읽는 법:** x50(h)를 h에 대해 그린 곡선은 생존 곡면 S(t, x0)의 50% 등고선이다. RC와 GT의 등고선이 겹치는지 보면 된다.
- **같은 원리로** 3.x의 Brier, reliability, 불확실성 분해도 모두 h마다 계산한다.

---

## 6. 앙상블 분포 지표 (2단계부터 사용)

### 6.1 CRPS `crps_ensemble`
- **정의:** CRPS = E|X − y| − ½ E|X − X'|. 앙상블 표본 X로 연속값 y(예: GT의 x50)를 예측할 때의 점수다. 단위는 y와 같고 낮을수록 좋다.
  - 정렬을 이용한 O(m log m) 계산. `fair=True`는 m(m−1) 추정량이다.
- **쓰는 곳:** x50 같은 **연속값** 비교에만 쓴다. x0별 비율에는 쓰지 않는다 (GT 비율이 유한 표본이라 해석이 애매하기 때문).
- **selftest:** brute force와 일치한다.

### 6.2 불확실성 분해 `uncertainty_decomposition`
- **정의 (전체 분산의 법칙, Bernoulli 결과에 대해 정확히 성립):**
  `p̄(1 − p̄) = E_j[p_j(1 − p_j)] + Var_j(p_j)`
  - 전체 = aleatoric(시스템 확률성) + epistemic(멤버 간 불일치)
  - 엔트로피 버전: H(p̄) = E_j H(p_j) + MI (Depeweg et al. 2018 계열).
- **반복 잡음 보정:** p_j 자체도 R번 반복으로 얻은 추정치라서 Var_j(p̂_j)에 binomial 잡음 p(1−p)/R이 섞인다.
  - `epistemic_corrected = Var_j(p̂) − mean_j p̂(1 − p̂)/(R − 1)`
  - selftest: 모든 멤버가 똑같은 경우 보정 전 0.0088 → 보정 후 −0.0008 (≈ 0).
- **가설:** epistemic은 x50 근처에서 가장 클 것이다.

### 6.3 밴드 coverage `band_coverage`
- 진짜 확률(독립 GT seed로 추정)이 앙상블 5–95% 밴드 안에 들어가는 x0의 비율이다. 보조 지표다. 독립성 가정에 주의한다 (Hamill 2001).

### 6.4 이탈 종류: clip 포화 vs 동역학적 이탈 (`run_phase2_single_node.py`)
- **배경:** RC rollout은 `np.clip`으로 범위가 제한된다. 궤적이 발산하면 clip 경계에 붙어 일정한 값을 낸다. 그 값이 −2.0 이하이면 이벤트 검출기(국소 최소 ≤ −2.0)가 곧바로 "이탈"로 잡는다.
- **정의:**
  - 멤버별로 각 시행이 **처음 clip 경계에 닿은 시각** t_clip을 기록한다.
  - RC 이탈 시각 T와 비교해 t_clip ≤ T + 0.5 s이면 **포화 이탈(saturation)**, 아니면 **동역학적 이탈(dynamic)** 로 나눈다.
  - 그 밖에 non-finite 시행 수와 clip에 머문 시간 비율도 함께 기록한다.
- **왜:** 원래 이벤트 정의로 계산한 지표는 그대로 보고하되, RC의 이탈이 "발작 같은 동역학"인지 "수치 발산"인지를 같이 밝혀야 해석이 정직해진다. 이것이 계획했던 health 분류기(EESN R0b의 L0 blowup에 해당)의 최소 버전이다.

---

## 7. 지표 우선순위 요약

| 순위 | 지표 | 묻는 것 |
|---|---|---|
| 1 | 시행 단위 Brier / BSS + Murphy 분해 + reliability | H 안 이탈 확률이 정확하고 보정되었나 |
| 2 | IBS + 시간별 BS(t) + D-calibration + KM 겹쳐 그리기 | **언제** 이탈하는지의 분포가 맞나 |
| 3 | x50 / s 분포 (멤버별 vs GT bootstrap), CRPS(x50) | 임계 위치와 전이 날카로움 |
| 4 | 불확실성 분해, coverage | 모델 불확실성이 어디서 큰가, 밴드가 정직한가 |
