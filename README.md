# Video Style XML Marker/Split Project

이 프로젝트는 컷편집된 영상에서 DINOv2-small 임베딩을 추출해 원본 비디오의 KEEP/REVIEW 구간을 찾고, Premiere Pro import용 FCP7 XML을 생성합니다.

기존 방식은 **편집본 영상 전체의 임베딩 스타일 프로파일**만 만들었습니다.  
이번 버전은 추가로 **원본↔편집본 페어 데이터**를 사용할 수 있습니다.

---

## 1. 추천 구조: 원본↔편집본 페어 기반

원본 비디오 하나에서 여러 편집본이 나오는 구조를 지원합니다. 또한 편집본 하나가 여러 원본을 섞어 만든 경우도 지원합니다.

파일명 규칙은 단순합니다.

```text
raw_videos/
  A001.mp4
  A002.mp4

edited_videos/
  A001_cut_01.mp4
  A001_highlight.mp4
  A001_shortform_v2.mp4
  A002_cut_01.mp4
```

즉, **편집본 파일명 안에 원본 파일명 stem이 포함**되어 있으면 자동으로 같은 그룹으로 묶습니다. 편집본 파일명 안에 원본 stem이 여러 개 들어 있으면 multi-source edit로 인식해서 각 원본의 학습에도 동시에 사용합니다.

예:

```text
원본: A001.mp4
편집본: A001_cut_01.mp4, A001_highlight.mp4, A001_shortform_v2.mp4

원본: A001.mp4, A002.mp4
편집본: A001_A002_mix_edit.mp4  # A001과 A002 양쪽 그룹에 자동 배정
```

내부적으로는 이름순 정렬 후 다음처럼 처리합니다.

1. 원본 비디오별로 편집본들을 그룹화
2. 원본 프레임 임베딩 추출
3. 같은 원본에서 나온 편집본 프레임 임베딩 추출
4. 원본 프레임이 편집본 프레임과 얼마나 가까운지 cosine similarity로 역매칭
5. 실제 편집본에 사용된 것으로 보이는 원본 구간을 KEEP segment로 생성
6. source-specific prototype과 global positive prototype을 저장

---

## 2. 페어 프로파일 생성

```bat
build_paired_profile.bat
```

주요 설정:

```bat
set RAW_DIR=raw_videos
set EDITED_DIR=edited_videos
set PROFILE_OUT=paired_profile.pkl
set SAMPLE_FPS=2.0
set MATCH_THR=0.78
set SMOOTH_SEC=2
set MIN_KEEP_SEC=1
set MERGE_GAP_SEC=1
```

`MATCH_THR`가 낮으면 더 많은 구간이 KEEP으로 잡히고, 높으면 더 보수적으로 잡힙니다.

초기 권장값:

```text
MATCH_THR=0.76 ~ 0.82
SAMPLE_FPS=2.0
SMOOTH_SEC=1~3
MIN_KEEP_SEC=1
```


---

## 3. LightGBM KEEP 모델 학습

페어 프로파일을 만든 뒤, 그 안의 KEEP segment를 pseudo label로 사용해서 LightGBM 모델을 학습할 수 있습니다.

```bat
build_lgbm_profile.bat
```

처리 흐름:

```text
paired_profile.pkl의 KEEP segment
→ 원본 프레임별 positive/negative label 생성
→ DINOv2 임베딩 + prototype score + temporal feature 생성
→ LightGBM binary classifier 학습
→ lgbm_profile.pkl 저장
```

기본값:

```bat
set PAIRED_PROFILE=paired_profile.pkl
set LGBM_PROFILE_OUT=lgbm_profile.pkl
set NEGATIVE_RATIO=4
set NUM_LEAVES=31
set LEARNING_RATE=0.03
set N_ESTIMATORS=400
```

원본이 17개 정도일 때는 DINOv2 자체를 fine-tuning하지 않고, 임베딩은 고정한 뒤 LightGBM만 학습하는 방식을 권장합니다. negative가 너무 많아지는 것을 막기 위해 positive 1개당 negative를 기본 4개까지만 샘플링합니다.

---

## 4. XML 생성

```bat
infer_xml_batch.bat
```

기본값은 `lgbm_profile.pkl`을 사용합니다. `paired_profile.pkl`과 기존 `style_profile.pkl`도 계속 사용할 수 있습니다.

```bat
set PROFILE_PATH=lgbm_profile.pkl
set USE_CACHED_PAIRED_SEGMENTS=0
```

`PROFILE_PATH=lgbm_profile.pkl`이면 LightGBM이 KEEP 확률을 예측합니다. 기본 threshold는 확률 기준입니다.

```bat
set KEEP_THR=0.55
set REVIEW_THR=0.35
```

`USE_CACHED_PAIRED_SEGMENTS=1`이면, `build_paired_profile.py`에서 이미 원본↔편집본 매칭으로 찾은 segment를 그대로 XML에 씁니다. 이 옵션은 `paired_profile.pkl` 확인용으로 유용합니다. LightGBM 추론을 할 때는 `0`으로 두세요.

`USE_CACHED_PAIRED_SEGMENTS=0`이면, 선택한 profile로 다시 추론합니다. `lgbm_profile.pkl`이면 LightGBM, `paired_profile.pkl`이면 source-specific/global prototype을 사용합니다.

---

## 5. 기존 edited-only 방식

기존처럼 편집본만 가지고 스타일 프로파일을 만들 수도 있습니다.

```bat
build_profile.bat
```

이 방식은 `style_profile.pkl`을 만들고, 원본에서 편집본 스타일과 비슷한 구간을 찾습니다.

페어 데이터가 있다면 `build_paired_profile.bat` 사용을 권장합니다.

---

## 6. 출력 방식

`infer_xml_batch.py`는 입력 원본 비디오마다 동명의 `.xml`을 생성합니다.

예:

```text
raw_videos/
  A001.mp4
  A002.mp4

raw_videos_xml/
  A001.xml
  A002.xml
```

각 XML은 원본 비디오 파일을 참조하고, 새 sequence를 생성합니다. `<pathurl>`에는 `file://` URL이 아니라 비디오의 절대 경로를 그대로 씁니다. 추천 KEEP/REVIEW 구간의 시작/끝 지점마다 clipitem이 분할되어 있어서 Premiere에 import하면 원본 전체가 추천 경계 기준으로 split된 상태로 보입니다.

---

## 7. 주의점

- 편집본 파일명에는 원본 파일명 stem이 들어가야 합니다.
- 원본 이름이 너무 짧으면 잘못 매칭될 수 있습니다. 예: `A.mp4`보다는 `A001_camera1.mp4`가 안전합니다.
- 여러 원본 이름이 동시에 매칭되면 multi-source edit로 보고 각 원본 그룹에 동시에 배정합니다. 단, 실제 positive 구간은 원본↔편집본 similarity threshold를 넘은 부분만 남습니다.
- 편집본에 자막, 효과, 줌, 색보정이 강하게 들어가도 DINOv2 임베딩 기반이라 어느 정도 견디지만, 화면이 완전히 바뀌는 트랜지션/인서트 컷은 오매칭될 수 있습니다.
- 경계가 둔하면 `SAMPLE_FPS`를 올리고 `SMOOTH_SEC`를 낮추세요.
