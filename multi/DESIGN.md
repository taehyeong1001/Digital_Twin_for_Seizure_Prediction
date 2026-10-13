# 엄밀한 멀티노드(3노드) SCDT — 사전 등록 프로토콜

> 2026-09-29: 최종본은 학습 λ를 13개(0.005 간격)로 늘린 버전이다. 변경점은 `DESIGN_ADDENDUM_dense_training.md`에 있다. 아래 표의 학습 λ 5개는 원래 사전 등록 내용이며, 이전 결과와 함께 `archive/q05_20records_2026-09-24/`에 보존했다.

저수지를 학습하기 전에 작성. SCDT_7 본문 Sec. 2.3과 Supplementary Material을 따르고,
`../single`의 분리 규칙을 그대로 계승한다. 네트워크·저수지·innovation·BO 범위는
`MultiNode_Epileptor`의 3노드 SCDT(`experiment_three_node_scdt*.py`)를 참고했다.

## 3노드 선행 실험과의 차이 (엄밀화)

| 항목 | 기존 3노드 SCDT | 이번 strict multi |
|---|---|---|
| static 입력 | x0 → 9개 통계 Ridge 보정 맵의 값 | 각 학습 기록에서 **직접 측정한** 통계 2개 |
| 은닉 파라미터 | 3차원 x0 벡터 (최대 x0 위험 좌표) | 1차원 drift λ (x0_i = λ + offset_i) |
| 학습 데이터 | 일부 실험은 postcritical 포함 | **정상(event-free) 기록만** |
| 하이퍼파라미터 선택 | 검증 생존/이탈 라벨 사용 | **정상 검증 기록만** 사용 |
| 임계값 | x0 축 생존곡선 비교 | 통계 경로 sweep의 50% crossing (논문 Eq. 25) |

## Ground truth

- Proix-2017 Epileptor 3노드, 삼각형 연결성 [[0,1,.45],[1,0,.70],[.45,.70,0]], G = 0.7,
  x2·y2 가산 잡음(분산 0.0025). `CODE/paper_epileptor_core.py`는 `../single`과 동일.
- x0_i = λ + [0, −0.05, −0.10]. 관측량: 노드별 x2−x1, 32 Hz.
- 정상 이탈 사건(GT·SCDT 공통): 5 s blanking 후 어느 노드든 ≤ −0.5.
- λ는 기록 생성과 검증축 정렬에만 사용.

## 데이터 (서로 다른 seed)

| 코호트 | λ | 기록 | 용도 |
|---|---|---|---|
| 학습 | −2.13, −2.115, −2.10, −2.085, −2.07 (수집 순서) | 각 4 × 160 s | readout, 통계, 경로 |
| 정상 검증 | −2.1225, −2.1075, −2.0925, −2.0775 | 각 2 × 160 s | BO (정상만) |
| 보정 | −2.14 … −2.045 (11개) | 각 4 × 160 s | 검증축 정렬만 |
| GT 경계 | −2.07 … −1.99 (0.0025 간격) | 30 × 200 s | 잠금 후 참조 |

모든 기록은 600 s target transient 후 160 s 보존, 이탈이 있으면 폐기·재생성.
GT 경계: λ = −2.07에서 600 s 준비 후 목표 λ로 전환, 200 s horizon.
탐색 스캔(fixed λ): λ ≤ −2.045 무사건, −2.035 이상 전부 발작 (λc ≈ −2.04).

## 조건 통계 (d_s = 2), λ를 쓰지 않는 선택 규칙

후보(노드 평균): mean, σ, lag-1 AC, lag-8 AC, PE(order 3, 16 Hz), 5% quantile, IQR, |상관|.
1. 선행 좌표 = 학습 기록의 regime 간/regime 내 분산비 F 최대 통계
2. 두 번째 좌표 = 수준(level) 계열(mean, q05)을 제외한 F 최대 통계 → 항상 변동성 좌표 포함
3. 경로: 20개 학습 기록에 s2 = a s1 + b 최소제곱
4. drift 방향 = 수집 순서에 따른 선행 통계의 추세 부호 (λ 값 미사용)

## 저수지

1500-unit sparse ESN (density .003), 3 동적 입력 + 2 static(학습 z-score),
readout [φ(r), u, 1] (홀수 항 제곱), ridge, stride 4, next/delta 모드,
학습 잔차 블록 innovation (3노드 block BO와 동일 개념).

## 모델 선택 (정상만)

GP-EI BO 40회 (시작점: 3노드 stable-only BO trial 25, robust block trial 152 + 일반 2개).
목적함수 = 통계 재현 scaled MAE + |응답 기울기 − 1| + 8 × 거짓 이탈률 + 4 × clipping.
GT 이탈·생존 정보는 들어가지 않음.

## 잠금 후 시험

1. 선행 통계를 경로를 따라 학습 범위 안쪽에서 바깥 2배 폭까지 36점 sweep, 점당 30 rollout × 200 s,
   warm-up은 가장 가까운 학습 regime의 20 s 창.
2. SCDT 임계값 = 첫 하향 50% 생존 crossing, 이항 bootstrap 95% CI.
3. 저수지 seed 5개로 반복 (하이퍼파라미터 고정).
4. 음성 대조: static 0으로 둔 동일 구조.
5. GT 경계 50% crossing(λ) → 보정 회귀로 통계축에 사상(검증 전용).

사전 성공 기준: 주 seed의 SCDT crossing이 존재하고 외삽 영역에 있으며, GT 사상 임계값이
seed 범위 또는 bootstrap CI 안에 들어온다. 보수적 = SCDT crossing이 GT보다 학습 쪽에 위치.
