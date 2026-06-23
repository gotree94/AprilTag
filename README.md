
# 마커 종류

## 1. 아루코 마커 (ArUco Marker)
* 특징: 검은색 테두리 안에 흑백의 격자(Grid) 패턴으로 이루어진 2차원 마커입니다.
* 용도: QR코드가 정보를 담는 '문서'의 느낌이라면, 아루코 마커는 3차원 공간상의 '이정표' 역할을 합니다. <br> 카메라가 마커를 인식하면 고유 ID는 물론, 마커와 카메라 사이의 거리, 기울기, 3차원 좌표를 <br> 매우 적은 연산량으로  빠르게 계산할 수 있어 로봇 공학에서 애용됩니다.
* 관련 기술: OpenCV 아루코 마커 가이드에서 인식 원리와 활용 방법을 확인할 수 있습니다.

## 2. 에이프릴태그 (AprilTag)
* 특징: 미시간 대학에서 개발한 시각적 기준 마커 시스템으로, 아루코와 유사하게 흑백 사각형 패턴을 사용합니다.
* 용도: 조명 변화, 모션 블러(흔들림) 등 시각적 환경이 좋지 않은 곳에서도 오인식률이 매우 낮아, <br> 실제 산업 현장의 자율주행 물류 로봇(AGV)이나 드론의 위치 측정, AR(증강현실)에 가장 널리 쓰이고 있습니다.

## 3. ARToolKit 및 ARTag
* 특징: 초창기 AR(증강현실) 구현을 위해 널리 사용된 마커 기술입니다.
* 용도: 카메라에 마커가 인식되면 그 좌표와 각도를 바탕으로 가상의 3D 객체(캐릭터, 가구 등)를 <br> 모니터 화면 속 마커 위에 자연스럽게 띄우는 용도로 활용되었습니다.

---

# AprilTag 기반 차량 위치 추적 시스템

4개의 AprilTag(tag36h11, ID 0-3)를 이용하여 차량의 바닥 좌표를 실시간 추적하는 프로그램입니다.  
**Stage 2**에서는 딥러닝 없이 전통적 CV 방식으로 장난감 자동차 2대를 추가 인식합니다.

---

## 1. 개발 환경

| 항목 | 사양 |
|------|------|
| OS | Windows 11 |
| Python | 3.13.9 |
| 카메라 | 2m 높이, 중앙 하단에서 하향 경사 촬영 |
| 입력 영상 | 1920×1080, 30fps, 331프레임 (약 11초) |
| AprilTag family | tag36h11 (태그 간 거리 103cm) |

### 의존성

```
pip install opencv-python numpy pupil-apriltags pillow
```

- `opencv-python` — 영상 입출력, GUI, 이미지 전처리 (CLAHE, threshold)
- `numpy` — 배열 연산, 좌표 변환
- `pupil-apriltags` — AprilTag 검출 (C 바인딩)
- `pillow` — AprilTag PNG 생성

---

## 2. AprilTag 생성

`generate_apriltags.py`로 ID 0~10까지 11개의 PNG를 생성했습니다.

**태그 구조 (10×10 셀):**

```
[흰색] [검정] [6×6 데이터] [검정] [흰색]
```

| 레이어 | 두께 | 색상 | 역할 |
|--------|------|------|------|
| 바깥쪽 | 1셀 | 흰색 | 검출 임계값 기준 |
| 안쪽 | 1셀 | 검정 | 태그 경계 검출 |
| 데이터 | 6×6 | 흑백 | ID 정보 |

**생성된 태그:**

| ID 0 | ID 1 | ID 2 | ID 3 | ID 4 |
|:----:|:----:|:----:|:----:|:----:|
| ![](tag36h11_0.png) | ![](tag36h11_1.png) | ![](tag36h11_2.png) | ![](tag36h11_3.png) | ![](tag36h11_4.png) |

| ID 5 | ID 6 | ID 7 | ID 8 | ID 9 | ID 10 |
|:----:|:----:|:----:|:----:|:----:|:-----:|
| ![](tag36h11_5.png) | ![](tag36h11_6.png) | ![](tag36h11_7.png) | ![](tag36h11_8.png) | ![](tag36h11_9.png) | ![](tag36h11_10.png) |

---

## 3. 시스템 구성

### 3.1 태그 배치 (ID 0-3)

```
ID 0 (0, 0) ───── 103cm ──── ID 1 (1030, 0)
     │                            │
     │     [ Car 1 ]              │
     │                            │
     │              [ Car 2 ]     │
     │                            │
ID 3 (0, 1030) ─── 103cm ──── ID 2 (1030, 1030)
```

- 물리 좌표계: ID 0을 원점으로 한 직교 좌표계 (mm)
- 태그 간 거리: 1030mm (사용자 제공)

### 3.2 데이터 흐름

```
cam.mp4 → 그레이스케일 → CLAHE 전처리
                              ↓
                    AprilTag 검출 (3단계)
                              ↓
                    태그 중심점 → 차량 중심
                              ↓
                    호모그래피 → 바닥 좌표(mm)
                              ↓
                    Dark Threshold + ROI
                              ↓
                    자동차 검출 (Contour)
                              ↓
                    오버레이 → output.mp4
                    CSV 기록 → position.csv
                    실시간 GUI
```

---

## 4. 핵심 알고리즘

### 4.1 다단계 AprilTag 검출 (`detect_tags`)

빛 반사, 그림자, 부분 가림에 강인하게 검출하기 위해 3단계로 구성했습니다.

**Pass 1 — Raw 검출**

원본 그레이스케일 이미지에서 1차 검출합니다. 대부분의 프레임에서 이 단계만으로 4개 태그 전부 검출됩니다.

**Pass 2 — CLAHE 검출**

Pass 1에서 누락된 태그가 있으면 CLAHE(Contrast Limited Adaptive Histogram Equalization)를 적용한 이미지에서 재검출합니다. CLAHE는 국소 영역별로 대비를 정규화하여 반사광으로 인한 명암 왜곡을 보정합니다.

클립 한계(`clipLimit=3.0`)로 과도한 증폭을 억제하고, 8×8 타일로 국소 영역을 분할합니다.

**Pass 3 — ROI 추적 검출**

이전 프레임에서 검출된 위치 주변 300×300px ROI에서 Raw/CLAHE 각각 재검출하여 여전히 누락된 태그를 찾습니다. ROI가 전체 프레임의 약 2.5%에 불과해 계산 부하가 낮습니다.

### 4.2 차량 위치 추정 (`compute_car_center`)

검출된 모든 태그 중심점의 산술 평균을 차량 중심으로 추정합니다. 1~3개 태그만 검출되어도 위치 추정이 가능합니다.

### 4.3 바닥 좌표 변환 (`calibrate_homography` / `estimate_floor_position`)

4개 태그의 이미지 좌표와 물리 좌표 쌍으로 호모그래피 행렬 `H`를 계산합니다.

호모그래피는 첫 번째로 모든 태그가 검출된 프레임에서 단 1회 계산하여 고정 사용합니다. 이를 통해 카메라 각도와 렌즈 왜곡이 보정된 실제 바닥 좌표(mm)를 얻습니다.

### 4.4 자동차 검출 (`detect_cars`) — Stage 2

딥러닝 없이 전통적 CV 방식으로 장난감 자동차를 인식합니다.

**원리:**

1. Grayscale 이미지에 `THRESH_BINARY_INV`를 적용하여 어두운 객체(자동차)를 분리
2. ROI 설정 파일(`roi_config.json`)에서 각 차량의 관심 영역을 로드
3. 각 ROI 내에서 가장 큰 Contour를 찾아 Bounding Box 생성
4. 최소 면적(`min_area`) 필터로 노이즈 제거

**ROI 설정:**

사용자가 `select_roi.py`로 영상 첫 프레임에서 직접 ROI를 그리면 JSON 파일로 저장됩니다. `track_car.py` 실행 시 이 파일을 자동 로드합니다.

```python
# 파라미터
_CAR_THRESH = 80              # 어두움 기준 (0-255)
_CAR_DEFAULT_MIN_AREA = 400   # 최소 blob 면적
```

### 4.5 Detector 파라미터 튜닝

```python
apriltag.Detector(
    families="tag36h11",
    quad_decimate=1.0,   # 원본 해상도 유지
    quad_sigma=0.4,      # 경계 검출 노이즈 감소
    refine_edges=1,      # 경계 정밀 보정
    decode_sharpening=0.5, # 디코딩 선명도 향상
)
```

---

## 5. 개선 과정

### 5.1 문제: ID 0 반사광 미검출 (Stage 1)

Pass 1(Raw)만으로는 ID 0이 특정 구간에서 빛 반사로 인해 검출되지 않는 현상 발생. 검출률 약 50%.

**원인 분석:** 바닥의 반사 소재가 태그 영역에 국소적 명도 왜곡을 유발하여 AprilTag quad 검출 실패.

**해결:**
- CLAHE 전처리 도입 (clipLimit 3.0, tileGridSize 8×8)
- 다단계 검출 파이프라인 구성 (Raw → CLAHE → ROI)
- ROI 추적으로 직전 위치 기반 재검색

**결과:** ID 0 검출률 50% → 99% 향상 (331프레임 중 328프레임 검출)

### 5.2 문제: ID 1 검출 저하 (Stage 1)

CLAHE 전처리를 모든 프레임에 무조건 적용했을 때, ID 1의 검출률이 떨어지는 역효과 발생.

**원인:** CLAHE가 조명이 양호한 영역에서 오히려 노이즈를 증폭하여 quad 검출을 방해.

**해결:**
- Raw 우선 검출 → CLAHE는 누락된 태그에만 선택적 적용
- 최종 결과 Raw/CLAHE 병합

**결과:** 모든 ID 99~100% 검출률 달성

### 5.3 문제: opencv-python GUI 미지원 (Stage 1)

`opencv-python-headless`와 `opencv-python`이 함께 설치되어 GUI 기능(`cv2.imshow`)이 비활성화됨.

**해결:**
- `pip uninstall opencv-python-headless`로 충돌 제거
- `pip install --force-reinstall opencv-python`으로 재설치
- 자동 GUI 감지 로직 추가 (headless 환경에서도 `--headless` 플래그 없이 실행 가능)

### 5.4 문제: 변수명 충돌 (Stage 1)

프레임 높이 변수 `H`와 호모그래피 행렬 변수 `H`가 충돌하여 `VideoWriter`에 잘못된 `frameSize` 전달.

**해결:** 프레임 높이 변수를 `H` → `FRAME_H`로 변경.

### 5.5 문제: Car 2 검출 실패 — 배경 차분 & HSV (Stage 2)

초기에는 배경 차분(median background subtraction)과 HSV 색상 임계값을 시도했으나 실패.

**원인:**
- 배경 차분: 바닥 아래쪽 strip(y>908)의 아티팩트(고주파 노이즈, frequency>0.50)가 그림자와 병합되어 500px 너비의 거대 blob 형성
- HSV 임계값: Car 2의 Saturation 56-61 (바닥 8-15)은 구분 가능하나, Value 56-61 (바닥 200) 차이가 너무 커서 3×3 morphology 이후 color cue가 소멸

**해결:**
- Dark threshold (`THRESH_BINARY_INV`, value < 80) + ROI 전환
- `select_roi.py`로 사용자가 직접 ROI를 그리도록 개선
- `min_area` 필터로 false positive 제거

**결과:** 두 차량 모두 전 프레임(331/331) 검출 성공

---

## 6. 실행 방법

```bash
# 1) AprilTag 생성 (필요시)
python generate_apriltags.py

# 2) ROI 설정 (최초 1회)
python select_roi.py
#   - 첫 프레임에서 각 차를 드래그하여 ROI 지정
#   - s: 저장, u: 취소, c: 초기화, q: 종료

# 3) 추적 실행
python track_car.py              # GUI 모드
python track_car.py --headless   # Headless 모드
```

### 출력 파일

| 파일 | 설명 |
|------|------|
| `output.mp4` | 검출 결과가 오버레이된 영상 |
| `position.csv` | 프레임별 태그 좌표 및 차량 위치 데이터 |
| `roi_config.json` | 사용자가 그린 ROI 설정 (자동 로드) |

### CSV 컬럼

| 컬럼 | 설명 |
|------|------|
| `frame` | 프레임 번호 |
| `tag{0..3}_x`, `tag{0..3}_y` | 각 태그 중심 픽셀 좌표 (미검출 시 빈칸) |
| `car_x_mm`, `car_y_mm` | 차량 중심 바닥 좌표 (mm) |
| `car_x_px`, `car_y_px` | 차량 중심 픽셀 좌표 |
| `{label}_x, _y, _w, _h` | ROI별 차량 bounding box (ROI label 기준, 동적 생성) |

---

## 7. 화면 오버레이 정보

실시간 영상에 다음 정보가 표시됩니다:

- **초록색 사각형** — 각 AprilTag 경계
- **노란색 점** — 각 태그 중심점
- **주황색 연결선** — 태그 간 거리 (실제 103cm + 픽셀 거리)
- **빨간색 화살표** — 태그 → 차량 중심 방향 및 상대 좌표(mm)
- **빨간색 원** — 태그 기반 차량 중심점
- **자동차 박스** — 각 ROI 내 검출된 자동차 (색상 구분)
- **중앙 좌표** — 각 자동차 박스 중심의 `(x, y)` 픽셀 좌표
- **태그 거리** — 자동차 박스 하단에 각 태그까지의 거리(mm)
- **상단 정보창** — 차량 위치(mm), 픽셀 좌표, FPS

---

## 8. 사용자 설정

### 태그 배치

실제 태그 배치에 맞게 `WORLD_PTS_MM` 값을 수정하세요.

```python
WORLD_PTS_MM = np.float32([
    [0,         0],          # ID 0 — 좌상단
    [1030,      0],          # ID 1 — 우상단
    [1030,      1030],       # ID 2 — 우하단
    [0,         1030],       # ID 3 — 좌하단
])
```

### 차량 검출 파라미터

```python
_CAR_THRESH = 80              # 어두움 임계값 (낮을수록 더 어두운 객체만 검출)
_CAR_DEFAULT_MIN_AREA = 400   # 최소 blob 면적 (노이즈 제거)
```

각 ROI별로 `min_area`를 개별 설정하려면 `roi_config.json`에 추가:
```json
{"label": "Car 1", "x": 623, "y": 231, "w": 207, "h": 191, "min_area": 500}
```

`VIDEO_PATH`를 원하는 입력 파일 경로로 변경하면 다른 영상에도 적용 가능합니다.

---

## 9. 성능

### Stage 2 종합 성능

| 지표 | 값 |
|------|------|
| 평균 처리 속도 | ~4.8 fps (1920×1080) |
| ID 0 검출률 | 99% (328/331) |
| ID 1 검출률 | 100% (331/331) |
| ID 2 검출률 | 99% (328/331) |
| ID 3 검출률 | 100% (331/331) |
| Car 1 검출률 | 100% (331/331) |
| Car 2 검출률 | 100% (331/331) |

---

## 10. 파일 구조

```
opencode/
├── cam.mp4                    # 입력 영상
├── track_car.py               # 메인 추적 프로그램
├── generate_apriltags.py      # AprilTag PNG 생성기
├── select_roi.py              # ROI 선택 도구
├── roi_config.json            # ROI 설정 파일 (select_roi.py로 생성)
├── output.mp4                 # 결과 영상
├── position.csv               # 프레임별 좌표 데이터
├── tag36h11_0.png ... _10.png # AprilTag 이미지 (ID 0-10)
└── README.md
```
