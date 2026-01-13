# auto_segment

클릭 기반 UI로 변수 가공부터 Demand Space 시각화까지 수행하는 세그먼트 분석 툴입니다. 드래그 앤 드롭으로 세그먼트를 조정하고, AI 어시스턴트로 즉각적인 해석·디버깅을 제공합니다.

![세그먼트 자동화 v2](https://raw.githubusercontent.com/jay-lay-down/auto_segment/main/source/%EC%84%B8%EA%B7%B8%EB%A8%BC%ED%8A%B8%20%EC%9E%90%EB%8F%99%ED%99%94_v2.png)
![AI Assistant](https://github.com/jay-lay-down/auto_segment/blob/main/source/rag%EC%82%AC%EC%9A%A9.png)

- 예시 데이터: https://github.com/jay-lay-down/auto_segment/blob/main/assets/FAKE_SEGMENT.xlsx  
- 프로그램 다운로드: https://drive.google.com/uc?id=1AM6l7GJ6M72bnEtBa_xuRqlPfoh_Cedk&export=download

---

## 📦 설치 및 실행 (Quickstart)
```bash
pip install PyQt6 pyqtgraph pandas numpy scikit-learn openpyxl requests
python app.py
```

---

## 🚀 사용 방법 (How to use)
분석 흐름을 따라가며 탭을 순서대로 진행하는 것을 권장합니다.

1) **데이터 불러오기**  
   - [데이터] 탭 → Raw Data 엑셀(.xlsx) 선택.  
   - 워크북에 `RECODE*` 시트가 있으면 모두 병합해 [RECODE 매핑] 탭에 표시합니다(QUESTION/CODE/NAME, 선택적 QUESTION_KR/NAME_KR).

2) **RECODE 매핑 편집**  
   - [RECODE 매핑] 탭: 테이블 직접 수정 또는 “Reload RECODE sheets”로 원본 시트를 다시 읽기.  
   - 라디오 버튼으로 “원문(QUESTION/NAME)” vs “한글(QUESTION_KR/NAME_KR)” 라벨 모드를 선택하여 이후 분석에 사용할 라벨을 지정.

3) **변수 만들기 (Group/Compose) — 핵심**  
   - [GROUP/COMPOSE] 탭.  
   - Binary recode(두 값으로 A/B 그룹), Grouping map(라벨 매핑), Compose(여러 `_seg` 결합 → 최종 세그 키).  
   - 예: `gender_seg|age_seg` 형태로 Demand Space용 세그먼트 키를 생성.

4) **요인 분석 (PCA/EFA) [선택]**  
   - [Factor Analysis] 탭에서 변수 선택 후 PCA 또는 EFA 실행.  
   - 요인/주성분 점수는 세그먼트 유사도 계산 시 선택적으로 추가됩니다.

5) **드라이버 분석 (Decision Tree) [선택]**  
   - [의사결정트리] 탭. 종속변수(예: factor score)와 독립변수를 지정 후 실행.  
   - 개선도 피벗, Split 상세, 그룹 추천을 확인하며 RECODE 라벨 모드에 따라 코드→라벨 매핑이 자동 적용됩니다.

6) **Demand Space (시각화 & 튜닝)**  
   - [Demand Space] 탭.  
   - Segments-as-points: `_seg` 조합을 점 1개로 표현, 타깃 분포 피벗 기반 거리 → PCA/MDS 좌표 → K-Means/Ward 클러스터.  
   - Variables-as-points: 변수 간 상관 기반 거리 → PCA/MDS → K-Means/Ward.  
   - 드래그로 점/라벨 이동, 클러스터 병합, 자동 라벨 배치/초기화를 지원합니다.

7) **Export**  
   - [Export] 탭에서 전처리 데이터, PCA/트리 결과, Demand Space 좌표·클러스터, RECODE 등을 엑셀로 저장합니다.

8) **AI Assistant / RAG**  
   - OpenAI 또는 Gemini API 키 입력 후 데이터·코드 질의를 수행합니다. 모델명은 UI에서 선택하거나 직접 입력합니다.

---

## ⚙️ 원리 (How it works)

### 데이터 & RECODE 정규화
- 모든 `RECODE*` 시트를 병합하고 컬럼명을 QUESTION/CODE/NAME(+QUESTION_KR/NAME_KR)으로 정규화합니다.  
- 라벨 모드(원문/한글)에 따라 코드→라벨, QUESTION→QUESTION_KR 매핑을 선택 적용해 downstream 분석·UI에 반영합니다.  
- 변수 타입은 자동 감지 + 수동 지정(Numeric/Categorical)으로 관리합니다.

### 세그먼트 생성 파이프라인
1. Binary/그룹핑/조합 기능으로 `_seg` 변수 생성.  
2. 선택한 `_seg`를 구분자로 연결해 세그먼트 키를 만들고(`sep`), Min N 필터로 희소 조합을 제거.  
3. 필요한 경우 요인 점수 평균을 세그먼트 피처에 추가합니다.

### Demand Space 좌표 계산
- **Segments-as-points**:  
  - 타깃×세그먼트 피벗을 n-count로 생성 → 열 단위 정규화(세그먼트 분포 벡터).  
  - Euclidean 거리 → PCA(특징 차원 ≥2) 또는 MDS로 2D 좌표.  
  - K-Means(n_init=10) 또는 Ward H-Clust로 클러스터 지정.  
  - 세그먼트별 n을 함께 보관해 점 메타로 표시.
- **Variables-as-points**:  
  - 타깃×변수 피벗을 n-count로 생성 → 열 정규화 후 거리 → PCA/MDS → K-Means/Ward.  
  - 각 변수의 응답 수(n)를 메타로 보관합니다.

### 인터랙션 & 편집
- 점 자유 이동, 라벨 이동(Shift), 클러스터 병합(라벨 드래그), 라벨 자동 배치/리셋을 지원합니다.  
- 편집된 좌표·클러스터가 Export에 그대로 반영됩니다.

### Export 파이프라인
- 00_Metadata, 01_Data(전처리), 02_RECODE, Factor Loadings/Scores, Decision Tree 결과, Demand Space 좌표·클러스터·프로필 시트를 생성합니다.

### AI & RAG
- OpenAI/Gemini API로 데이터/분석 질의응답.  
- `build_code_rag.py`를 실행하면 코드베이스를 Chroma로 색인해 CLI 챗봇에서 파일 경로를 인용한 답변을 제공합니다.

---

### 🔑 AI 기능 설정 (API Key)
AI 어시스턴트를 쓰려면 OpenAI 또는 Gemini API 키가 필요합니다. 실행 후 AI Assistant (RAG) 탭에서 입력하세요.

> OpenAI 기본 모델: `gpt-4o-mini` (UI 기본값), 고성능: `gpt-4o`.  
> Gemini 기본 모델: `gemini-3-pro-preview` (UI 기본값). 권장 모델: `gemini-3-pro-preview`, `gemini-3.5-pro-preview`, `gemini-3.5-pro-preview-0409`, `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-1.5-flash-001`. UI 라벨에 괄호/`models/`가 있어도 자동 정규화합니다.  
> 보안: 저장소에 키는 포함되지 않습니다. 실행 시 UI 또는 환경변수(`OPENAI_API_KEY`, `GEMINI_API_KEY`)로 입력하세요.

### 🔎 로컬 코드 RAG 어시스턴트
1. `pip install -r requirements-rag.txt`  
2. `export OPENAI_API_KEY="sk-..."` (PowerShell: `$env:OPENAI_API_KEY="sk-..."`)  
3. `python build_code_rag.py --project-root . --rebuild`  
   - `chroma_db/`에 인덱스 저장, 이후 `--rebuild` 없이 재사용 가능.  
   - `--extensions`로 `.py` 외 확장자 추가 가능.  
4. 프롬프트에 질문을 입력하면 코드 경로를 인용해 답변합니다.

### 💡 활용 팁
- [데이터] → [Factor] → [Compose] → [Demand Space] 순으로 진행하면 빠르게 스토리라인을 만들 수 있습니다.  
- 자동 클러스터가 마음에 들지 않으면 라벨을 드래그해 병합하고, “Auto-Arrange Labels”로 가독성을 높이세요.  
- `_seg` 네이밍을 일관되게 유지하면 이후 피벗/조합/Export가 편해집니다.

### 🪟 PyInstaller 빌드 팁 (Windows)
```bash
python scripts/build_windows.py --name auto_seg --entry app.py
```
- 경로·이름을 단순화해 `Invalid argument` 오류를 예방합니다. 필요 시 `--dist`, `--build`, `--extra`로 폴더와 옵션을 지정하세요.

### 🔄 GitHub 동기화 안내
원격이 설정되어 있지 않습니다. GitHub에 반영하려면 원격을 추가한 뒤 `work` 브랜치(또는 사용 중 브랜치)를 직접 push 해야 합니다.

### 👨‍💻 Author: Jihee Cho (https://github.com/jay-lay-down)

See `docs/ANALYSIS_SPEC.md` for clustering and similarity specifications.
