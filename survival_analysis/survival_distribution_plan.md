# Survival rate를 "분포"로 예측하고 평가하기 — 아이디어와 진행 계획

> 작성: 2026-10-12. 대상: 이 프로젝트를 같이 하는 사람, 그리고 나중의 나.
> 이 문서는 (1) 왜 survival rate를 점 추정이 아닌 분포로 다뤄야 하는지, (2) 관련 분야가 이를 어떻게 처리하는지와 근거 논문,
> (3) 이 폴더(`eeg_survival`)에서 무엇을 어떤 순서로 구현할지를 순서대로 정리한다.
> 논문 근거는 웹에서 다시 확인했고, 각 항목에 **확인 수준**(원문/초록 확인 vs 2차 출처만 확인)을 표시했다.
>
> **중간 보고서 (2026-10-13):** [docs/REPORT_survival_distribution_2026-10-13.md](docs/REPORT_survival_distribution_2026-10-13.md)
>
> **진행 문서 (2026-10-12 추가):**
> - [docs/01_current_pipeline.md](docs/01_current_pipeline.md): 기존 코드가 실제로 계산하는 것
> - [docs/02_metrics.md](docs/02_metrics.md): 지표 정의서
> - [docs/03_progress_log.md](docs/03_progress_log.md): 결정과 결과 기록
> - [docs/04_fair_comparison.md](docs/04_fair_comparison.md): 기존 프레임워크 vs 우리 실험 — 차이와 공정한 비교 설계 (2026-10-13)
> - [docs/05_next_experiments.md](docs/05_next_experiments.md): 다음 실험 설계 — 생존 두 가지 보기, 통계량 검증, 시간 상관 가설 (2026-10-13)
> - 코드: `survival_dist/`
>
> 결정 변경: refill은 하지 않는다 (§3.4는 이미 반영됨). 1단계 결과에 따라 시간 축 지표의 비중을 높였다 (03 참고).

---

## 0. 한 줄 요약

지금은 reservoir 하나가 x0마다 survival rate **하나**를 낸다. 이를 **seed만 다른 reservoir 앙상블**로 바꿔 x0마다
survival 확률의 **분포**(모델 불확실성)와 이탈 시점 T의 **분포**(생존 곡선)를 예측한다. 평가는 생존분석과 확률 예보 분야의 표준 도구로 한다.
- **주 지표:** 시행 단위 Brier score와 reliability diagram
- **곡선 요약:** x50과 기울기의 분포
- **시간 차원:** KM 곡선과 integrated Brier score
- **불확실성 분해:** 멤버 간 / 멤버 내 분산

EESN(Julia) 저장소에서는 필요한 모듈만 Python으로 옮겨 온다. 이때 static 조건(통계 좌표)을 받을 수 있게 고친다.

---

## 1. 출발점: 지금 무엇을 하고 있나

(자세한 내용은 `final_code/README.md`)

- **GT:** Epileptor ODE (single node 또는 3-node), 노이즈 포함.
- **RC (SCDT):** 동적 입력(궤적) + **static 입력(관측 prefix의 통계: mean / std / lag-1 AC 등)** 을 받는 sparse reservoir.
  - 구현: `MultiOutputStaticRC`, `BlockInnovationRC`.
  - rollout 동안 static 조건을 목표 x0에 해당하는 값으로 고정한다.
  - 잔차를 block 단위로 다시 뽑아(block innovation) 노이즈로 넣는다.
- **정상 이탈:** 예측 시작 5초 이후 어느 노드든 신호가 `-0.5` 이하로 내려가는 것.
- **Survival rate:** x0별로 `1 − mean(event)` (`grouped_survival`). horizon 끝 시점 하나의 비율이다.

**문제:** 이 값은 **reservoir 하나**에서 나온 **점 추정**이다.
- 같은 하이퍼파라미터로 seed만 바꾼 reservoir가 다른 survival 곡선을 낼 수 있다. 그 폭(모델 불확실성)을 지금은 모른다.
- GT 곡선도 유한한 seed 수로 얻은 추정치다. "RC 곡선이 GT 점을 지나는가"는 엄밀한 비교가 아니다.
- 이탈 **여부**만 보고 **시점**을 버리고 있다.

---

## 2. 문헌 검토: 단계별로

### 2.1 "통계/파라미터를 reservoir에 넣어 전이를 예측한다"는 방향 자체의 근거

| 주장 | 근거 | 확인 수준 |
|---|---|---|
| reservoir에 **파라미터 입력 채널**을 추가하고 정상 영역에서만 학습해도 임계 전이 지점을 예측할 수 있다 | Kong, Fan, Grebogi & Lai (2021), *Phys. Rev. Research* 3, 013090 | 초록 확인 |
| 같은 논문에서, 붕괴 전 **transient lifetime의 분포와 평균**까지 추정했다 | 위와 같음 | 초록 확인 |
| 파라미터를 모른 채 비정상(non-stationary) 시계열에서 tipping과 tipping 이후 동역학을 외삽한다 | Patel & Ott, arXiv:2207.00521 (*Chaos* 게재로 알려짐) | arXiv 확인, 게재지는 미확인 |
| 실제 EEG에서 **variance와 autocorrelation 증가(critical slowing down)** 가 발작 위험과 연관된다 | Maturana et al. (2020), *Nat. Commun.* 11, 2172 | 초록 확인 |

**이 프로젝트와의 연결:**
- 우리 SCDT의 static 입력은 Kong et al.의 "parameter channel"을 일반화한 것으로 볼 수 있다. 파라미터 자체 대신 **관측 통계**를 넣는다.
- 그 통계로 mean, std, AC1을 고른 것은 Maturana et al.의 CSD 결과와 맞는다.
- Kong et al.이 이미 **이탈까지 걸리는 시간의 분포**를 reservoir로 추정했다. 그래서 "이탈 시점 분포를 예측 대상으로 삼는다"는 방향에는 선행 사례가 있다.

### 2.2 Survival rate는 통계적으로 무엇인가

정의상 survival rate는 **P(T > H | x0)** 다. T는 정상 이탈 시점, H는 horizon이다. 관련 분야는 이 확률을 세 가지 층위에서 다룬다.

**(a) 시행 하나는 Bernoulli 실험이다. 비율의 표본 불확실성은 binomial로 처리한다.**
- dynamical systems의 **basin stability**(Menck et al. 2013, *Nat. Phys.*)가 구조적으로 거의 같다. 초기조건을 무작위로 뽑아 attractor로 돌아오는 비율을 Monte Carlo로 추정하는 방법이다.
  - 이 분야에서는 그 추정치의 표준오차로 √(S(1−S)/N)을 쓴다 (N회 반복한 Bernoulli 실험의 분산). [2차 출처 확인: arXiv:1711.03857, bSTAB 논문 등. Menck 원문의 해당 식은 직접 확인하지 못함]
  - 같은 문헌에서 S가 0 또는 1에 가까울 때 이 식이 0을 내는 문제를 지적한다.
- 비율의 신뢰구간에는 Wald(√(p(1−p)/n)) 대신 **Wilson이나 Jeffreys 구간**을 권장한다 (Brown, Cai & DasGupta 2001, *Stat. Sci.*). [초록 확인]
  - Jeffreys 구간 = Beta(k+½, n−k+½)의 분위수다. 즉 Bayesian 사후분포와 바로 연결된다.

**(b) 이탈 시점 T 자체가 분포다. 생존분석과 확률 동역학으로 다룬다.**
- **생존분석:** S(t) = P(T > t)를 Kaplan–Meier로 추정하고, Greenwood 공식으로 구간을 붙인다.
  - horizon 안에 이탈하지 않은 시행은 **우측 중도절단**이다. 교과서적인 표준 방법이다.
- **확률 동역학:** 노이즈로 인한 탈출(Kramers형)이나 transient chaos의 생존확률은 대체로 **지수적으로 감소**한다. 그 지수가 탈출률 κ이고, 평균 첫 통과 시간(MFPT) ≈ 1/κ이다.
  - 근거: transient chaos 문헌 (예: Altmann & Tél 2009, *PRE* 79, 016204), 약한 노이즈 극한의 escape-time 분포 문헌. [2차 출처 확인]
  - 예외: KAM 섬 근처의 sticky orbit이 있으면 power law가 될 수 있다.
- **우리에게 주는 의미:** x0마다 S(t) 곡선 또는 κ(x0)를 비교하면 RC가 "이탈 여부는 맞히지만 시점은 틀리는" 경우를 잡아낼 수 있다.

**(c) 모델 자체의 불확실성 (epistemic)**
- **앙상블:** 독립적으로 학습한 여러 모델의 예측을 섞는 단순한 방법이 잘 보정된 예측 불확실성을 준다 (Lakshminarayanan, Pritzel & Blundell 2017, NeurIPS "Deep Ensembles"). [초록 확인]
  - 이 논문은 학습과 평가 기준으로 Brier score 같은 proper scoring rule을 쓴다.
- **ESN 앙상블로 불확실성을 정량화한 선례:** McDermott & Wikle (2017), "An Ensemble Quadratic Echo State Network for Nonlinear Spatio-Temporal Forecasting" (arXiv:1708.05094). [초록 확인]
  - 그들의 quadratic readout은 우리의 짝수/홀수 노드 제곱 readout과 같은 계열이다.
- **불확실성 분해:** 예측 불확실성을 epistemic(모델)과 aleatoric(데이터 노이즈)으로 나눈다 (Depeweg et al. 2018, ICML). [초록 확인. 정확한 식은 원문에서 확인 필요]

### 2.3 x0에 따른 곡선 = dose–response 곡선

"x0가 커질수록 survival이 떨어지는 곡선"은 독성학의 **용량–반응 곡선**과 구조가 같다. 용량에 따라 생존 비율이 떨어지는 곡선이다.
- **표준 방법:** probit 또는 logistic 회귀로 S(x0) = 1 − F((x0 − x50)/s)를 적합한다.
  - 요약값은 **x50 (= LD50)** 과 기울기 s다.
  - x50 추정치는 두 계수의 비율이라서, 신뢰구간에 **Fieller 정리**(또는 delta method, bootstrap)를 쓴다 (Finney의 probit analysis, SAS PROBIT 문서). [2차 출처 확인]
  - 주의: x50이 시험한 용량 범위에서 멀면 Fieller와 delta 구간이 매우 넓어진다는 보고가 있다. x0 격자를 x50 근처에 촘촘히 두어야 한다.
- **장점:** 곡선 전체를 비교하는 대신 **임계점 위치(x50)** 와 **전이의 날카로움(s)** 이라는, 물리적으로 해석되는 두 숫자로 비교할 수 있다. 우리 multi-node 실험이 이미 x50을 쓰고 있어 바로 이어진다.

### 2.4 확률 예측을 평가하는 표준 도구

| 목적 | 도구 | 근거 | 확인 수준 |
|---|---|---|---|
| 전체 원칙 | **보정(calibration)을 만족하는 범위에서 예측을 최대한 날카롭게(sharpness)** 한다. 평가는 proper scoring rule로 한다 | Gneiting, Balabdaoui & Raftery (2007), *JRSS-B*; Gneiting & Raftery (2007), *JASA* | 초록과 2차 출처 확인 |
| 이진 사건 확률 | **Brier score**, Brier skill score | 기상·의료 표준 | — |
| Brier 분해 | BS = REL − RES + UNC (보정 − 판별 + 사건 자체의 불확실성) | Murphy (1973) | 2차 출처 확인 |
| 보정을 그림으로 | **reliability diagram + consistency bar**(resampling) | Bröcker & Smith (2007), *Weather Forecast.* 22, 651 | 초록 확인 |
| 앙상블 보정 | rank histogram. 단, 평평하다고 해서 보정되었다는 보장은 없다 | Hamill (2001), *Mon. Wea. Rev.* 129, 550 | 초록 확인 |
| 분포 예측 보정 | PIT 히스토그램 (보정되었다면 Uniform) | Gneiting et al. (2007) | 초록 확인 |
| 시간에 따른 생존 예측 | **time-dependent Brier score, integrated Brier score (IBS)** | Graf, Schmoor, Sauerbrei & Schumacher (1999), *Stat. Med.* 18, 2529 | 서지와 정의 확인 (scikit-survival 문서) |
| 개별 생존 분포 보정 | **D-calibration**: 각 시행에서 S_pred(T_i)가 Uniform인지 본다 (생존분석판 PIT) | Haider, Hoehn, Davis & Greiner (2020), *JMLR* 21(85) | 초록 확인 |

### 2.5 seizure forecasting 분야의 관행 (나중에 실제 EEG로 갈 때)

- 이 분야는 "예측한다/못한다"에서 **확률적 위험 예보**로 옮겨 가고 있다. 리뷰: Kuhlmann et al. (2018), *Nat. Rev. Neurol.* 14, 618. [서지 확인. 구체적인 지표 권고는 원문 확인 필요]
- **Proix et al. (2021), *Lancet Neurol.* 20, 127:** 장기 iEEG 175명의 multiday 주기로 다음 날 발작 위험을 예보했다. **Brier skill score**로 평가했다 (중앙값 BSS 0.23 / 0.13). [요약 기사로 확인]
- Karoly et al. (2020), *Epilepsia*: 발작 가능성의 주기(circadian/multiday)를 이용한 예보. [서지 확인]

**결론:** Brier, BSS, reliability를 주 지표로 쓰면 합성 실험과 나중의 실제 EEG 실험, 그리고 기존 seizure forecasting 연구를 같은 잣대로 비교할 수 있다.

---

## 3. 그래서 이렇게 설계한다

### 3.1 표기

- x0: 목표 파라미터 (multi-node는 3차원. 그림의 x축은 기존처럼 max(x0) 등을 쓴다)
- 멤버 j = 1..N: 하이퍼파라미터는 BO로 고정하고 **reservoir seed만 다르게** 한다.
- 멤버마다 innovation 노이즈 반복 r = 1..R → 이탈 시점 T_{j,r}(x0)를 얻는다. horizon까지 이탈이 없으면 중도절단.
- p_j(x0) = 멤버 j의 이탈 확률 추정 (= 1 − survival)
- 앙상블 예측: p̄(x0) = mean_j p_j(x0)
- GT: Epileptor seed b = 1..B → 이탈 여부 o_b(x0) ∈ {0,1}와 이탈 시점

### 3.2 무엇을 "분포"라고 부르나 (세 가지를 모두 보고한다)

1. **survival 확률의 분포 (모델 불확실성):** {p_j(x0)}_j. 그림에는 x0별 밴드(5–95%)로 그린다.
2. **이탈 시점의 분포:** S(t | x0). 앙상블 혼합의 KM 곡선과, 멤버별 곡선의 밴드로 그린다.
3. **GT 쪽 불확실성:** Jeffreys 사후분포 Beta(k+½, B−k+½)와 KM의 Greenwood 구간.

**불확실성 분해.** Bernoulli 결과에 대해 전체 분산의 법칙이 정확하게 성립한다.

```
p̄(1−p̄)  =  E_j[ p_j(1−p_j) ]  +  Var_j( p_j )
 (전체)       (aleatoric: 시스템 확률성)   (epistemic: 멤버 간 불일치)
```

- 엔트로피 버전도 있다: H(p̄) = E_j[H(p_j)] + MI (Depeweg et al. 계열).
- **주의:** p_j도 R번 반복으로 얻은 추정치라서 Var_j(p̂_j)에는 binomial 잡음이 섞여 있다. 그래서 다음처럼 보정한다.
  `Var_between ≈ Var_j(p̂_j) − mean_j[ p̂_j(1−p̂_j) ] / (R−1)`
- **가설 (논문 포인트 후보):** epistemic 성분은 x50 근처에서 가장 클 것이다.

### 3.3 평가 지표 (우선순위)

1. **시행 단위 Brier score와 BSS.**
   - GT 시행 (x0, b) 하나하나의 실제 결과 o_b를 RC 앙상블 예측 p̄(x0)와 비교한다.
   - 이렇게 하면 GT가 유한 표본이라는 문제가 자동으로 처리된다. proper scoring rule이다.
   - 비교 기준선:
     - (i) 기존 단일 RC의 p
     - (ii) climatology (모든 x0에서 같은 평균 이탈률)
     - (iii) 이론상 상한: 독립 GT seed로 추정한 p_GT(x0)
2. **reliability diagram** (Bröcker & Smith의 consistency bar 포함)과 Murphy 분해 REL / RES / UNC.
3. **x50과 기울기의 분포.**
   - 멤버마다 logistic을 적합해 {x50_j, s_j}를 얻는다.
   - GT는 seed bootstrap으로 x50 분포를 만든다.
   - 두 분포를 겹쳐 그리고, x50의 CRPS도 계산한다.
4. **시간 차원.**
   - x0 bin별로 KM 곡선을 그린다: RC 앙상블 밴드 vs GT의 Greenwood 구간.
   - **IBS**를 계산한다. 우리 데이터의 중도절단은 horizon 끝 한 시점(administrative censoring)뿐이라서 IPCW 가중이 사실상 단순해진다.
   - **D-calibration**을 확인한다.
   - (선택) κ(x0) 비교.
5. **x0별 coverage:** GT의 p_GT(x0)가 앙상블 5–95% 밴드에 들어가는 비율. 보조 지표로만 쓴다.

> 처음에 생각했던 "x0별 survival rate에 CRPS를 매긴다"는 GT 쪽이 유한 표본이라 해석이 애매하다.
> 그래서 주 지표는 시행 단위 Brier/reliability로 바꿨다. CRPS는 x50처럼 연속값을 비교할 때만 쓴다.

### 3.4 일부러 하지 않는 것과 주의점

- **refill(나쁜 멤버 교체)은 하지 않는다.** 앙상블은 seed만 다른 멤버를 있는 그대로 쓴다. 선택 편향을 없애기 위해서다.
- **resync(EESN의 핵심 기법)도 쓰지 않는다.** resync는 "건강하지 않은" 멤버를 덮어쓴다. 그런데 여기서는 정상 이탈 자체가 예측하려는 사건이라, 이탈을 지워 survival을 과대추정하게 된다.
- **health 분류기는 이탈 전 구간의 수치 붕괴 판정에만 쓴다.** blowup, fixed point, periodic lock 같은 경우다. 목적은 "수치적으로 망가진 rollout"과 "물리적 정상 이탈"을 구분해 보고하는 것이다.
- **seed 앙상블은 모델 불확실성의 일부만 잡는다.** 하이퍼파라미터가 고정되어 있기 때문이다. 필요하면 나중에 BO 상위 k개 설정을 섞은 앙상블과 비교한다.
- **rank histogram이나 coverage의 유의성 검정은 독립 표본을 가정한다** (Hamill 2001). 같은 seed를 공유하는 시행끼리는 묶어서 다룬다.

---

## 4. 구현 계획 (이 폴더에서, Python)

`final_code/`는 수정하지 않고 import만 한다. 새 코드는 `survival_dist/`(가칭)에 둔다.

### 4.1 EESN에서 가져올 모듈

| EESN 원본 (Julia) | 새 모듈 | 바꿀 점 |
|---|---|---|
| `ensemble/training.jl` `train_ensemble_checkpointed` | `ensemble.py` | seed만 다른 `BlockInnovationRC` N개를 만든다. 멤버별로 `w_out`과 residual을 npz로 저장한다. 이미 있으면 건너뛰고, config fingerprint로 확인한다. **static 조건을 받는 rollout**을 그대로 쓴다 |
| `R_experiments/R0/R0c_metrics_and_plots.jl` `kaplan_meier`, `km_half_life` | `survival.py` | 이탈 시점과 중도절단 플래그를 입력으로 받는다. Greenwood 구간과 Jeffreys 구간을 추가한다 |
| `R_experiments/R0/forecast_dsr_metrics.jl` CRPS / spread / skill | `dist_metrics.py` | Brier, BSS, Murphy 분해, reliability와 consistency bar, IBS, D-calibration, x50 적합과 CRPS, 불확실성 분해를 넣는다 |
| `R_experiments/R0/R0b_health_taxonomy.jl` (L0 blowup / L1 fixed_point / L2 periodic_lock / L4 drift) | `health.py` | 긴 GT 정상 구간으로 보정한다. 이탈 전 구간에만 적용한다 |
| `R_experiments/R0/ensemble_diversity_metrics.jl` N_eff | `diversity.py` (선택) | 멤버들이 사실상 같은 궤적으로 붕괴하지 않았는지 확인한다 |

### 4.2 순서 (각 단계에 통과 조건이 있다)

1. **`survival.py` + `dist_metrics.py`**
   - 기존 단일 RC 결과와 GT에 적용한다.
   - 통과 조건: horizon 시점의 KM 값이 기존 `grouped_survival`과 정확히 같다. 기존 그림과 수치가 재현된다.
2. **`ensemble.py`** (single node, N=20부터 시작)
   - 통과 조건: 원래 seed 멤버의 결과가 기존 결과와 일치한다. N을 늘려도 p̄가 안정된다.
3. **`health.py`**
   - 통과 조건: 보정에 쓰지 않은 GT 정상 구간에서 false alarm이 약 1% 수준이다.
4. **single node 분포 평가 실험**
   - 3.3의 지표와 그림을 모두 만든다. 단일 RC와 climatology 기준선을 같이 넣는다.
5. **3-node로 확장**
   - x축은 기존과 같이 max(x0)를 쓴다. 3차원 x0에서 x50을 어떻게 정의할지 결정한다.
6. (선택) 하이퍼파라미터까지 섞은 앙상블 비교, κ(x0) 비교.
7. (나중) 실제 EEG: x0 대신 통계 기반 위험 좌표를, `-0.5` 대신 onset annotation을 쓴다. 지표는 그대로 쓴다.

---

## 5. 같이 정해야 할 질문

- **앙상블 크기 N과 멤버당 반복 R:** 계산 예산 안에서 epistemic 추정과 aleatoric 추정의 정밀도를 어떻게 나눌지. 일단 N=20, R=기존 반복 수로 시작한다.
- **GT seed 수 B:** reliability diagram의 consistency bar가 의미 있으려면 x0 격자 × B가 충분해야 한다.
- **3-node에서 x50의 정의:** max(x0)라는 1차원 요약으로 충분한지, 아니면 3차원 경계면으로 다뤄야 하는지.
- **D-calibration에서 중도절단 처리 방식:** Haider et al.의 방식을 그대로 쓸지.

---

## 참고문헌 (웹에서 확인한 링크)

- Kong, Fan, Grebogi & Lai (2021). Machine learning prediction of critical transition and system collapse. *Phys. Rev. Research* 3, 013090. https://arxiv.org/abs/2012.01545
- Patel & Ott. Using machine learning to anticipate tipping points and extrapolate to post-tipping dynamics of non-stationary dynamical systems. https://arxiv.org/abs/2207.00521
- Maturana et al. (2020). Critical slowing down as a biomarker for seizure susceptibility. *Nat. Commun.* 11, 2172. https://pmc.ncbi.nlm.nih.gov/articles/PMC7195436
- Menck, Heitzig, Marwan & Kurths (2013). How basin stability complements the linear-stability paradigm. *Nat. Phys.* 9, 89. (2차 설명) https://arxiv.org/pdf/1711.03857 , https://link.springer.com/article/10.1007/s11071-021-06786-5
- Brown, Cai & DasGupta (2001). Interval estimation for a binomial proportion. *Stat. Sci.* 16, 101. https://projecteuclid.org/euclid.ss/1009213286
- Altmann & Tél (2009). Poincaré recurrences and transient chaos in systems with leaks. *PRE* 79, 016204. https://arxiv.org/abs/0808.3785
- Lakshminarayanan, Pritzel & Blundell (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS. https://arxiv.org/abs/1612.01474
- McDermott & Wikle (2017). An ensemble quadratic echo state network for nonlinear spatio-temporal forecasting. https://arxiv.org/abs/1708.05094
- Depeweg et al. (2018). Decomposition of uncertainty in Bayesian deep learning. ICML. https://proceedings.mlr.press/v80/depeweg18a.html
- SAS PROBIT 문서 — Fieller 구간과 LD50. https://support.sas.com/documentation/cdl/en/statug/63347/HTML/default/statug_probit_sect026.htm
- Gneiting, Balabdaoui & Raftery (2007). Probabilistic forecasts, calibration and sharpness. *JRSS-B* 69, 243. https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jrssb.pdf
- Murphy (1973) Brier 분해 — 설명: Siegert, Variance estimation for Brier score decomposition. https://arxiv.org/abs/1303.6182
- Bröcker & Smith (2007). Increasing the reliability of reliability diagrams. *Weather Forecast.* 22, 651. https://www.lse.ac.uk/CATS/Assets/PDFs/Publications/Papers/2007/73-IncreasingReliabilityDiagrams-2006-Brocker-Smith.pdf
- Hamill (2001). Interpretation of rank histograms for verifying ensemble forecasts. *Mon. Wea. Rev.* 129, 550. https://scores.readthedocs.io/en/stable/tutorials/Rank_Histogram.html
- Graf, Schmoor, Sauerbrei & Schumacher (1999). Assessment and comparison of prognostic classification schemes for survival data. *Stat. Med.* 18, 2529. (구현: https://scikit-survival.readthedocs.io/en/v0.15.0/api/generated/sksurv.metrics.integrated_brier_score.html)
- Haider, Hoehn, Davis & Greiner (2020). Effective ways to build and evaluate individual survival distributions. *JMLR* 21(85). https://jmlr.org/papers/v21/18-772.html
- Kuhlmann, Lehnertz, Richardson, Schelter & Zaveri (2018). Seizure prediction — ready for a new era. *Nat. Rev. Neurol.* 14, 618. https://aura.abdn.ac.uk/handle/2164/11941
- Proix et al. (2021). Forecasting seizure risk in adults with focal epilepsy. *Lancet Neurol.* 20, 127. https://boris.unibe.ch/149887
- Karoly et al. (2020). Forecasting cycles of seizure likelihood. *Epilepsia*. https://findanexpert.unimelb.edu.au/scholarlywork/1443041-forecasting-cycles-of-seizure-likelihood
