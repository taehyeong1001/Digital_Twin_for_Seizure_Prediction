# Survival rate를 분포로 평가하기 (SCDT-RC, Epileptor GT)

SCDT-RC가 내는 survival rate를 reservoir 하나의 점 추정이 아니라 **seed 앙상블의 분포**로 예측하고, 생존분석과 확률 예보 지표로 평가한 작업이다.

| 위치 | 내용 |
|---|---|
| `docs/` | 문서. 읽는 순서: 01 기존 로직 → 02 지표 → 03 진행 기록 → 04 공정 비교 → 05 후속 실험. 중간 보고서는 `REPORT_…` (⚠ 표시는 04/05에서 정정된 부분) |
| `survival_distribution_plan.md` | 전체 아이디어와 문헌 검토 |
| `survival_dist/` | 분석 코드와 결과(그림, 요약). 실행법은 `survival_dist/README.md` |
| `final_code/` | 분석이 import하는 SCDT 원본 코드 (수정하지 않음) |
| `final_figures/` | 원본 코드가 만든 그림 |

- 결과의 대용량 데이터(`*.npz`), 로그, 다중 노드 실행 사본은 올리지 않았다. `survival_dist/README.md`의 명령으로 다시 만들 수 있다.
- 환경: Python 3.12, `final_code/requirements.txt` + `scikit-survival`.
