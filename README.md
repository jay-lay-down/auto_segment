# auto_segment

복잡한 코딩 없이 클릭만으로 변수를 가공하고, Demand Space(=segmentation과 혼용하여 사용) 상에서 드래그 앤 드롭으로 세그먼트를 조정하여 최적의 타겟 그룹을 발굴할 수 있습니다.
![세그먼트 자동화 v2](https://raw.githubusercontent.com/jay-lay-down/auto_segment/main/source/%EC%84%B8%EA%B7%B8%EB%A8%BC%ED%8A%B8%20%EC%9E%90%EB%8F%99%ED%99%94_v2.png)
**AI 어시스턴트**가 탑재되어 분석 결과에 대한 즉각적인 해석과 오류 디버그, 분석 결과에 대한 질의응답 가능
![AI Assistant](https://github.com/jay-lay-down/auto_segment/blob/main/source/rag%EC%82%AC%EC%9A%A9.png)

## 데이터는 예시 데이터입니다. ##

📦 설치 및 실행
```python
pip install PyQt6 pyqtgraph pandas numpy scikit-learn openpyxl requests
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
타깃 분포 피벗(옵션): Segments-as-points에서 타깃 대비 세그먼트 분포 피벗을 만들고 그 비율로 세그먼트 유사도를 계산하는 R hclust 스타일을 선택할 수 있습니다.
좌표 및 클러스터링: PCA 또는 MDS 좌표계를 선택하고, K-Means 클러스터링(k개)을 실행합니다.
표시되는 점 개수에 대한 이해: Segments-as-points는 선택한 *_seg 조합(label)마다 **한 점**을 그립니다. `Min N` 필터로 걸러진 뒤 단 하나의 세그먼트만 남으면 화면에 점이 하나만 보일 수 있습니다. 원 데이터의 개별 응답자 포인트를 직접 찍는 모드는 아니며, 세그먼트별 집계 포인트(n 크기 가중치 포함)입니다. 반대로 Variables-as-points는 선택한 변수(컬럼)마다 **한 점**을 찍으므로, 최소 2개 이상의 변수를 선택하면 그 수만큼 점이 보입니다(데이터 행이 적어도 2개는 필요).
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

> OpenAI 기본 모델: `gpt-4o-mini` (UI 기본값), 고성능: `gpt-4o`.
>
> Gemini 기본 모델: `gemini-3-pro-preview` (UI 기본값). 지원·권장 모델 예시: `gemini-3-pro-preview`, `gemini-3.5-pro-preview`, `gemini-3.5-pro-preview-0409`, `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-1.5-flash-001`. UI에서 괄호가 포함된 라벨이나 `models/` 접두사가 붙어 있어도 앱이 자동으로 정규화해 올바른 엔드포인트로 요청합니다. 모델명이 잘못되면 404 안내 메시지와 함께 시도한 모델 목록을 표시합니다.

> 보안 참고: 이 저장소에는 어떤 API 키도 포함되어 있지 않습니다. 키는 실행 시 UI 입력란이나 환경변수(예: `OPENAI_API_KEY`, `GEMINI_API_KEY`)로 직접 넣어야 하며, 배포나 커밋 전에 키가 코드나 설정 파일에 남지 않았는지 한 번 더 확인해 주세요.

### 🔎 로컬 코드 RAG 어시스턴트
프로젝트 전체 코드를 벡터로 인덱싱해 "어떤 파일에서 이 함수가 정의됐지?" 같은 질문에 바로 응답하는 CLI 챗봇입니다.

1. 의존성 설치: `pip install -r requirements-rag.txt`
2. OpenAI 키 설정: `export OPENAI_API_KEY="sk-..."` (또는 PowerShell에서 `$env:OPENAI_API_KEY="sk-..."`)
3. 실행: `python build_code_rag.py --project-root . --rebuild`
   - `chroma_db/` 폴더에 코드 벡터 인덱스가 저장되며, 이후에는 `--rebuild` 없이 빠르게 재사용할 수 있습니다.
   - `--extensions` 옵션을 사용하면 `.py` 외 다른 확장자를 추가할 수 있습니다.
4. 터미널 프롬프트에 질문을 입력해 코드 위치와 함께 답변을 받아보세요.

### 💡 활용 팁
보통 [데이터 로드] → [PCA] (성향 파악) → [Compose] (인구통계 변수 조합) → [Demand Space] (Segments-as-points 모드) 순서로 진행하면 효율적입니다.
Demand Space에서 자동 클러스터링된 결과가 마음에 들지 않으면, 마우스로 라벨을 드래그하여 직관적으로 그룹을 합치세요.
_seg 변수를 만들 때 일관된 규칙(예: 변수명_seg)을 사용하면 관리가 편합니다. (툴에서 자동 처리해 줌)

### 🪟 Windows에서 PyInstaller 빌드 팁
PyInstaller로 `.exe`를 만들 때 경로가 길거나 특수문자가 포함되면 `Invalid argument` 오류가 날 수 있습니다. 아래 스크립트는 경로/이름을 정리한 뒤 빌드를 실행해 문제를 예방합니다.

1. **명령 예시**
   ```bash
   python scripts/build_windows.py --name auto_seg --entry app.py
   ```
   필요하면 `--dist`(출력 폴더), `--build`(작업 폴더), `--extra`(추가 PyInstaller 인자)를 덧붙일 수 있습니다.

2. **오류 대처**
   - dist/work 경로가 로컬 디스크인지 확인하세요(네트워크 드라이브/공유 폴더일 경우 권한 오류가 발생할 수 있음).
   - exe 이름에 공백·특수문자가 없는지 확인하고, 더 짧은 이름을 사용해 보세요.
   - 스크립트는 빌드 전에 경로 쓰기 가능 여부를 점검하고, 실패 시 안내 메시지를 출력합니다.

### 🔄 GitHub 동기화 안내

현재 이 작업 공간에는 Git 원격이 설정되어 있지 않습니다. 모든 커밋은 로컬 리포지토리에만 저장되며, GitHub에 반영하려면 원격 URL을 추가한 뒤 `work` 브랜치(또는 사용 중인 브랜치)를 직접 push 해야 합니다.

### 👨‍💻 Author: Jihee Cho (https://github.com/jay-lay-down)

See docs/ANALYSIS_SPEC.md for clustering and similarity specifications.
