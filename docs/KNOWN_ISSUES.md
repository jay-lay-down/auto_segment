# Known Issues

## `NameError: name 'feat_cols' is not defined` in Demand Space (해결 완료)
- 발생 위치: `Segments-as-points` 모드에서 `_run_demand_space` → `_build_segment_profiles` 호출 시.
- 원인: `_build_segment_profiles` 내부에서 피처 컬럼 목록을 `feature_cols`로 만들고 반환하면서, 호출부에서는 `feat_cols`라는 이름을 기대할 때 변수명이 불일치해 NameError가 발생했습니다.
- 현상: 함수가 `feat_cols`를 찾지 못해 즉시 NameError를 던지며, 이후 클러스터 계산이 진행되지 않습니다.
- 조치: `_build_segment_profiles`에서 반환 변수명을 `feat_cols`로 정리해 호출부와 통일했습니다. 이제 NameError 없이 세그먼트 프로파일을 계산할 수 있습니다.

## PyInstaller 빌드 로그에 `set_exe_build_timestamp` / `update_exe_pe_checksum` 경고가 반복됨
- 현상: PyInstaller로 Windows EXE를 만들 때 `OSError(22, 'Invalid argument')` 경고가 여러 번 출력되지만 마지막에 `Building EXE ... completed successfully` 메시지가 나타나 빌드는 완료됩니다.
- 원인: 리눅스/WSL 등 비-Windows 환경에서 PE 헤더의 타임스탬프와 체크섬을 덮어쓰는 단계가 플랫폼 제한으로 실패하면서 경고가 발생합니다.
- 영향: 경고만 발생하며 최종 EXE가 생성되면 치명적인 오류는 아닙니다. 실제 실행에 문제가 있으면 Windows 환경에서 동일한 명령으로 다시 빌드하거나, 경로에 특수 문자가 없는지 확인하세요.
