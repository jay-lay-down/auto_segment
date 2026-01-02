# Known Issues

## `NameError: name 'feat_cols' is not defined` in Demand Space (해결 완료)
- 발생 위치: `Segments-as-points` 모드에서 `_run_demand_space` → `_build_segment_profiles` 호출 시.
- 원인: `_build_segment_profiles` 내부에서 피처 컬럼 목록을 `feature_cols`로 만들고 반환하면서, 호출부에서는 `feat_cols`라는 이름을 기대할 때 변수명이 불일치해 NameError가 발생했습니다.
- 현상: 함수가 `feat_cols`를 찾지 못해 즉시 NameError를 던지며, 이후 클러스터 계산이 진행되지 않습니다.
- 조치: `_build_segment_profiles`에서 반환 변수명을 `feat_cols`로 정리해 호출부와 통일했습니다. 이제 NameError 없이 세그먼트 프로파일을 계산할 수 있습니다.
