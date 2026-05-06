# Video Style XML Marker/Split Project

이 프로젝트는 컷편집된 영상들에서 DINOv2-small 임베딩 스타일 프로파일을 만들고, 원본 비디오 디렉토리를 추론해서 비디오마다 Premiere Pro import용 FCP7 XML을 생성합니다.

## 출력 방식

`infer_xml_batch.py`는 입력 원본 비디오마다 동명의 `.xml`을 생성합니다.

예:

```text
raw_videos/
  A.mp4
  B.mp4

raw_videos_xml/
  A.xml
  A.json
  B.xml
  B.json
  _summary.json
```

각 XML은 원본 비디오 파일을 참조하고, 새 sequence를 생성합니다. 추천 KEEP/REVIEW 구간의 시작/끝 지점마다 clipitem이 분할되어 있어서 Premiere에 import하면 원본 전체가 추천 경계 기준으로 split된 상태로 보입니다. KEEP/REVIEW 구간은 sequence marker도 같이 들어갑니다.

## 사용 순서

1. `edited_videos/` 폴더를 만들고, 이미 컷편집된 영상들을 넣습니다.
2. `raw_videos/` 폴더를 만들고, 추론할 원본 비디오들을 넣습니다.
3. `setup_uv_env.bat` 실행
4. `build_profile.bat` 실행
5. `infer_xml_batch.bat` 실행
6. 생성된 `raw_videos_xml/*.xml`을 Premiere Pro에 import합니다.

## BAT에서 주로 수정할 값

### `build_profile.bat`

```bat
set EDITED_DIR=edited_videos
set SAMPLE_FPS=1.0
set INPUT_SIZE=224
set BATCH_SIZE=64
set CLUSTERS=64
```

### `infer_xml_batch.bat`

```bat
set INPUT_DIR=raw_videos
set KEEP_THR=0.74
set REVIEW_THR=0.68
set SMOOTH_SEC=7
set MIN_KEEP_SEC=3
set MERGE_GAP_SEC=2
```

값을 낮추면 더 많은 marker/split이 생기고, 값을 높이면 더 보수적으로 생성됩니다.

## Premiere에서의 결과

- XML을 import하면 원본 비디오를 참조하는 sequence가 생성됩니다.
- sequence 안의 클립은 KEEP/REVIEW 경계 기준으로 split되어 있습니다.
- sequence marker로 KEEP/REVIEW 구간 정보가 표시됩니다.
- 실제 원본 파일은 복사/재인코딩하지 않습니다.

## 참고

Premiere Pro의 기본 단축키에는 현재 playhead 위치에서 자르는 `Add Edit` / `Add Edit to All Tracks`가 있지만, marker 전체를 한 번에 add edit하는 공식 기본 명령은 없습니다. 이 프로젝트는 그 과정을 XML sequence로 우회해서, import 시점에 이미 split된 sequence를 만드는 방식입니다.
