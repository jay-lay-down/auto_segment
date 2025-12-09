# Known Issues

## `NameError: name 'feat_cols' is not defined` in Demand Space
- 발생 위치: `Segmentation-as-points` 모드에서 `_run_demand_space` → `_build_segment_profiles` 호출 시.
- 원인: `_build_segment_profiles` 내부에서 존재하지 않는 지역 변수 `feat_cols`를 참조하거나 반환하도록 작성된 이전 스크립트(`app16.py` 등) 때문입니다. 현재 로직은 `feature_cols`라는 이름으로 피처 목록을 만들기 때문에, 함수 내 코드와 반환값 이름이 불일치하면 NameError가 발생합니다.
- 현상: 함수가 `feat_cols`를 찾지 못해 즉시 NameError를 던지며, 이후 클러스터 계산이 진행되지 않습니다.
- 해결 방향: `_build_segment_profiles`에서 사용하는 피처 컬럼 변수명을 `feature_cols`와 일치하도록 정리하거나, 호출부/반환부에서 동일한 이름을 사용하도록 수정해야 합니다. (이번 커밋에서는 원인 설명만 제공합니다.)
