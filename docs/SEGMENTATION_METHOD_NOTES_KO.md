# 파이썬 세그멘테이션 계산 vs R 스크립트 계산 (요약)

## 파이썬 앱의 Segments-as-points 계산 흐름
- 선택한 세그 컬럼을 구분자로 연결해 모든 조합 라벨을 만든 뒤, 선택한 요인 점수(또는 타깃)를 **수치/더미 인코딩**합니다. 관련 구현은 `_build_segment_profiles`와 `_encode_features`에서 확인할 수 있습니다.【F:app.py†L3476-L3523】
- 인코딩된 프로파일을 표준화한 행렬 `Xz`를 **PCA(또는 입력이 부족하면 1-상관 기반 MDS)**로 2D 좌표화한 뒤, 그 좌표에 **Agglomerative(ward) / K-Means**를 적용해 클러스터 ID를 부여합니다.【F:app.py†L3400-L3456】
- 의사결정나무(decision tree)는 별도 탭에서 변수 중요도/베스트 스플릿을 보기 위한 도구이며, 세그먼트 클러스터링에 사용되지 않습니다.【F:app.py†L2380-L2467】

## 제공된 R 스크립트의 계산 흐름
- 타깃×세그 조합 빈도를 `dcast`로 피벗해 열별 합이 1이 되도록 정규화한 뒤, **유클리드 거리 행렬 → hclust(complete) → cutree(k)**로 클러스터를 자릅니다.【F:docs/R_SEGMENTATION_WORKFLOW_KO.md†L27-L53】
- 범주형을 더미로 변환하지 않고, 한국어 세그 라벨을 그대로 피벗 열로 사용해 거리 행렬을 계산합니다.【F:docs/R_SEGMENTATION_WORKFLOW_KO.md†L39-L46】
- 클러스터 후 결과를 CSV/엑셀로 내보내 재라벨(`seg_rev`, `seg_rev2` 등)을 붙이는 방식이며, 의사결정나무는 세그 군집화에 포함돼 있지 않습니다.【F:docs/R_SEGMENTATION_WORKFLOW_KO.md†L55-L65】

## 결론
- 질문하신 "세그 결과가 PCA와 dctree를 쓴 것인지"에 대해: **세그 클러스터는 파이썬에서 PCA/MDS 기반 좌표 + Ward/K-Means로 산출되며, dctree(의사결정나무) 결과는 클러스터링에 쓰이지 않습니다.**
- R 스크립트는 거리행렬 기반 hclust를 사용하며 더미 인코딩이나 의사결정나무를 세그 군집 계산에 넣지 않습니다.
- Segments-as-points 모드도 R과 동일하게 **세그 조합 전체를 거리행렬 기반 군집(complete/ward) 대상으로 삼는다**는 점만 유지하면 되며, UI에서 타깃/세그 선택 방식만 다를 뿐 클러스터 입력에 포함되는 조합 자체는 버려지지 않습니다.
