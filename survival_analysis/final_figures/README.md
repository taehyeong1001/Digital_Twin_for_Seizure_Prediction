# Final GT–RC figures

## Single node

- `single_node/single_node_GT_RC_inter_pre_post.png`
- `single_node/single_node_GT_RC_inter_pre_post.pdf`
- `single_node/single_node_survival_rate.png`
- `single_node/single_node_survival_rate.pdf`
- 구성: intercritical, precritical, postcritical 조건의 GT와 SCDT-RC 비교
- Rate: 고정된 target `x0`에 따른 GT와 SCDT-RC survival probability 비교
- 원본: `MultiNode_Epileptor/figures/RC/paper_single_node_scdt/gt_rc_inter_pre_post.*`
- Rate 원본: `MultiNode_Epileptor/figures/RC/paper_single_node_scdt/survival_rate.*`

## Multi node

- `multi_node/three_node_GT_RC_normal_departure.png`
- `multi_node/three_node_survival_rate.png`
- `multi_node/three_node_survival_rate_BO_comparison.png`
- 구성: 3-node 정상 및 departure/seizure 조건의 GT와 SCDT-RC 2×2 비교
- Rate: `max(x0,1, x0,2, x0,3)`에 따른 GT와 최종 SCDT-RC의 120초 survival fraction 비교
- BO comparison: rate 중심 BO와 peak 가중 BO의 survival curve 비교
- 선택 기준: peak 가중 BO 이후 robust confirmation을 통과한 최종 모델
- 원본: `MultiNode_Epileptor/figures/RC/three_node_scdt_block_peak_robust/gt_rc_2x2.png`
- Rate 원본: `MultiNode_Epileptor/figures/RC/three_node_scdt_block_peak_robust/survival_vs_boundary.png`
- BO comparison 원본: `MultiNode_Epileptor/figures/RC/three_node_scdt_block_peak_robust/survival_rate_comparison.png`
