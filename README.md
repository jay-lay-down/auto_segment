# auto_segment

복잡한 코딩 없이 클릭만으로 변수를 가공하고, Demand Space(=segmentation과 혼용하여 사용) 상에서 드래그 앤 드롭으로 세그먼트를 조정하여 최적의 타겟 그룹을 발굴할 수 있습니다.
![세그먼트 자동화 v2](https://raw.githubusercontent.com/jay-lay-down/auto_segment/main/source/%EC%84%B8%EA%B7%B8%EB%A8%BC%ED%8A%B8%20%EC%9E%90%EB%8F%99%ED%99%94_v2.png)
**AI 어시스턴트**가 탑재되어 분석 결과에 대한 즉각적인 해석과 오류 디버그, 질의응답이 가능합니다.
![AI Assistant](https://github.com/jay-lay-down/auto_segment/blob/main/source/rag%EC%82%AC%EC%9A%A9.png)


📦 설치 및 실행
```python
pip install PyQt6 pyqtgraph pandas numpy scikit-learn openpyxl request
```

# 실행
python app.py

## 🚀 사용 가이드 (Step-by-Step)
분석 흐름은 탭 순서대로 진행하는 것을 권장합니다.

### 1. 데이터 불러오기 (Load)
[데이터] 탭으로 이동합니다.
분석할 Raw Data 엑셀 파일(.xlsx)을 불러옵니다.
(선택사항) 엑셀 파일 내에 RECODE 시트(컬럼: QUESTION, CODE, NAME)가 있다면 자동으로 인식하여 [RECODE 매핑] 탭에 표시해 줍니다.

### 2. 변수 만들기 (Recode & Compose) ✨ 핵심
Demand Space에 점을 찍기 위해 필요한 세그먼트 변수(*_seg)를 만드는 단계입니다.
[GROUP/COMPOSE] 탭을 사용합니다.
빠른 이진 RECODE: 특정 컬럼의 값 2개를 골라 A/B 그룹으로 나누고 _seg 컬럼을 생성합니다. (예: 성별 → 남/녀)
그룹핑 매핑: 복잡한 척도형 변수를 구간별로 묶어 새로운 라벨을 붙입니다.
COMPOSE (조합): 위에서 만든 여러 _seg 변수들을 조합하여 최종 세그먼트 변수를 만듭니다.
예시: 성별_seg + 연령_seg → 성별|연령_seg (남자|20대, 여자|30대...)
Tip: 이 기능으로 만든 조합 변수가 Demand Space 분석의 핵심 기준이 됩니다.

### 3. 요인 분석 (PCA) (선택)
데이터의 차원을 축소하여 주요 경향성을 파악합니다.
[PCA] 탭에서 문항(변수)들을 체크합니다.
PCA 실행: 설정한 개수(k)만큼의 주성분(PCA1, PCA2...)이 생성되며, 이는 이후 단계의 Input으로 사용됩니다.
Loadings 미리보기: 어떤 변수가 어떤 요인에 묶였는지 직관적으로 확인할 수 있습니다.

### 4. 드라이버 분석 (Decision Tree & HCLUST)(선택)
어떤 변수가 세그먼트를 나누는 데 중요한지(Driver) 탐색합니다.
[의사결정트리] 탭으로 이동합니다.

Decision Tree Outputs 실행:
앞서 만든 PCA 점수들을 종속변수(dep)로, 나머지 문항들을 독립변수(ind)로 하여 모든 경우의 수를 탐색합니다.
Improve Pivot: 어떤 변수가 설명을 가장 잘하는지 점수화하여 보여줍니다.
HCLUST 실행: 변수들의 패턴이 유사한 것끼리 묶어줍니다. (변수 축소용)
[Split Results] 탭: 특정 변수가 어떻게 갈라지는지 상세 조건(불순도 차이 등)을 확인합니다.

### 5. Demand Space (시각화 및 튜닝) 🎨
분석 결과를 시각적으로 펼쳐 놓고, 마우스로 직접 수정하며 최종 그룹을 확정합니다.
[Demand Space] 탭으로 이동합니다.

모드 선택:
Segments-as-points: Step 2에서 조합한 세그먼트(예: 남자|20대)를 점으로 찍습니다.
Variables-as-points: 변수(컬럼) 자체를 점으로 찍어 변수 간 거리를 봅니다.
좌표 및 클러스터링: PCA 또는 MDS 좌표계를 선택하고, K-Means 클러스터링(k개)을 실행합니다.
인터랙티브 기능 (마우스 조작):
드래그 & 머지: 클러스터 라벨(중심점 이름)을 드래그해서 다른 클러스터 위에 놓으면 두 그룹이 합쳐집니다.
점 이동: 점 자유이동 ON 체크 시, 점을 드래그하여 좌표를 임의로 수정할 수 있습니다. (스토리텔링을 위한 배치 수정)
Shift + 라벨 드래그: 병합하지 않고 라벨 위치만 옮깁니다.
Shift + 클릭: 점을 다중 선택합니다.

### 6. 결과 저장 (Export)
[Export] 탭에서 파일 경로를 지정하고 저장 버튼을 누릅니다.
전처리된 데이터, PCA 결과, 트리 분석 결과, 그리고 수작업으로 조정한 Demand Space 좌표 및 클러스터 정보까지 모두 엑셀 시트로 분리되어 저장됩니다.

### 🔑 AI 기능 설정 (API Key)
AI 어시스턴트 기능을 사용하려면 OpenAI API Key가 필요합니다.
프로그램 실행 후 AI Assistant (RAG) 탭으로 이동합니다.
sk-로 시작하는 본인의 API 키를 입력합니다.
질문을 입력하고 Send를 누르면 데이터 기반의 답변을 받을 수 있습니다.

### 💡 활용 팁
보통 [데이터 로드] → [PCA] (성향 파악) → [Compose] (인구통계 변수 조합) → [Demand Space] (Segments-as-points 모드) 순서로 진행하면 효율적입니다.
Demand Space에서 자동 클러스터링된 결과가 마음에 들지 않으면, 마우스로 라벨을 드래그하여 직관적으로 그룹을 합치세요.
_seg 변수를 만들 때 일관된 규칙(예: 변수명_seg)을 사용하면 관리가 편합니다. (툴에서 자동 처리해 줌)

### 👨‍💻 Author: Jihee Cho (https://github.com/jay-lay-down)

