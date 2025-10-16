# YOLOv6 ONNX 변환 및 추론

이 레포지토리는 YOLOv6 모델을 ONNX 형식으로 변환하고, 변환된 모델을 사용하여 이미지와 비디오에서 객체 탐지를 수행하는 완전한 솔루션을 제공합니다.

## 🌟 주요 기능

- **YOLOv6 to ONNX 변환**: YOLOv6 PyTorch 모델을 ONNX 형식으로 자동 변환
- **이미지 객체 탐지**: 단일 이미지 또는 COCO 데이터셋 이미지 배치 처리
- **비디오 객체 탐지**: 실시간 시각화와 함께 여러 비디오 파일 배치 처리
- **크로스 플랫폼 지원**: Ubuntu, macOS, Windows에서 동작 (Docker 컨테이너 포함)
- **최적화된 성능**: 빠른 추론을 위한 ONNX Runtime 사용
- **상세한 분석**: 포괄적인 탐지 보고서 및 통계

## 📁 프로젝트 구조

```
yolov6-to-onnx-test/
├── yolov6_image.py          # COCO 샘플을 이용한 이미지 처리
├── yolov6_video.py          # 비디오 배치 처리
├── requirements.txt         # Python 의존성
├── input_videos/            # 입력 비디오 폴더
├── output_videos/           # 처리된 비디오 출력 폴더
├── coco_images/            # 다운로드된 COCO 샘플 이미지
├── detection_results/      # 이미지 탐지 결과
├── yolov6n.onnx            # 변환된 ONNX 모델
├── yolov6n.pt              # YOLOv6 PyTorch 가중치
└── README.md               # 이 파일
```

## 🚀 빠른 시작

### 요구사항

- Python 3.8 이상
- Git
- 인터넷 연결 (모델 및 데이터셋 다운로드용)

### 설치

1. **레포지토리 클론**:
   ```bash
   git clone https://github.com/jisunclaralee/yolov6-to-onnx-test.git
   cd yolov6-to-onnx-test
   ```

2. **의존성 설치**:
   ```bash
   pip install -r requirements.txt
   ```

   > **참고**: 스크립트는 시스템 의존성을 자동으로 설치하고 컨테이너 환경에서 OpenCV 호환성 문제를 처리합니다.

### 사용법

#### 1. COCO 샘플을 이용한 이미지 객체 탐지

```bash
python yolov6_image.py
```

이 스크립트는 다음을 수행합니다:
- 시스템 의존성 자동 설치
- YOLOv6 레포지토리 클론 및 설정
- 사전 훈련된 가중치 다운로드
- PyTorch 모델을 ONNX 형식으로 변환
- 5개의 다양한 COCO 데이터셋 샘플 이미지 다운로드
- 각 이미지에 대해 객체 탐지 수행
- 바운딩 박스가 포함된 주석 처리된 결과 저장
- 상세한 탐지 보고서 생성

**출력**: 
- `coco_images/`: 다운로드된 샘플 이미지
- `detection_results/`: 주석 처리된 이미지 및 JSON 요약
- `yolov6n.onnx`: 변환된 ONNX 모델

#### 2. 비디오 배치 처리

```bash
# 입력 디렉토리 생성 및 비디오 추가
mkdir -p input_videos
# 비디오 파일을 input_videos/에 복사

# 모든 비디오 처리 (ONNX 모델이 필요합니다)
python yolov6_video.py
```

**참고**: 비디오 처리를 위해서는 먼저 `yolov6_image.py`를 실행하여 ONNX 모델을 생성해야 합니다.

기능:
- `input_videos/` 디렉토리의 모든 비디오 처리
- 다양한 형식 지원: MP4, AVI, MOV, MKV, WMV, FLV, WebM
- 실시간 진행률 추적
- 상세한 처리 통계

**출력**:
- `output_videos/`: 객체 탐지 주석이 있는 처리된 비디오
- `processing_summary.json`: 포괄적인 처리 보고서

## 🎯 지원되는 객체 클래스

모델은 다음을 포함한 80개의 COCO 클래스를 탐지합니다:
- **사람**: person
- **차량**: bicycle, car, motorbike, aeroplane, bus, train, truck, boat
- **동물**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **물체**: bottle, chair, sofa, laptop, cell phone, book, scissors 등

## 📊 예시 출력

### 이미지 탐지 결과
```json
{
  "total_images": 5,
  "total_detections": 23,
  "results": {
    "coco_images/sample_1.jpg": [
      {
        "class": "cat",
        "confidence": 0.934,
        "bbox": [123, 45, 456, 234]
      }
    ]
  }
}
```

### 비디오 처리 통계
```json
{
  "total_videos_processed": 3,
  "total_detections": 1543,
  "total_processing_time": 120.5,
  "results": [
    {
      "input_file": "input_videos/sample.mp4",
      "frames_processed": 300,
      "total_detections": 450,
      "processing_time": 45.2
    }
  ]
}
```

## ⚙️ 설정

### 신뢰도 임계값
각 스크립트에서 탐지 신뢰도 임계값을 조정할 수 있습니다:

```python
CONFIDENCE_THRESHOLD = 0.25  # 기본값: 0.25 (25%)
```

### 모델 입력 크기
기본 입력 크기는 640x640 픽셀입니다. 변경하려면:

```python
# 변환 스크립트에서
--img-size 640  # 원하는 크기로 변경
```

## 🐳 Docker 지원

스크립트는 Docker 컨테이너에 최적화되어 있으며 자동으로 다음을 수행합니다:
- 시스템 의존성 설치 (`libgl1-mesa-glx`, `ffmpeg` 등)
- GUI 의존성을 피하기 위해 OpenCV headless 버전 사용
- 컨테이너 환경에서 권한 문제 처리

## 🔧 문제 해결

### 일반적인 문제

1. **OpenCV 임포트 오류**:
   ```
   ImportError: libGL.so.1: cannot open shared object file
   ```
   **해결책**: 스크립트가 자동으로 필요한 시스템 라이브러리를 설치하고 OpenCV headless 버전을 사용합니다.

2. **메모리 부족**:
   - 배치 크기나 비디오 해상도 줄이기
   - 비디오를 하나씩 처리
   - 낮은 신뢰도 임계값 사용

3. **Git 클론 타임아웃**:
   - 인터넷 연결 확인
   - 다른 DNS 서버 사용 시도
   - YOLOv6 레포지토리 수동 클론

### 성능 팁

- **CPU 최적화**: 스크립트는 최대 호환성을 위해 ONNX Runtime CPU 제공자를 사용합니다
- **메모리 사용량**: 메모리 사용량을 최소화하기 위해 프레임별로 처리됩니다
- **속도**: 더 빠른 처리를 위해 낮은 신뢰도 임계값을 사용하세요

## 📋 요구사항

### Python 의존성
전체 목록은 `requirements.txt`를 참조하세요:
- `numpy>=1.26.4`
- `onnxruntime>=1.17.3`
- `opencv-python-headless>=4.9.0.80`
- `torch` 및 `torchvision`
- 진행률 표시를 위한 `tqdm`
- 다운로드를 위한 `requests`

### 시스템 의존성 (자동 설치)
- `libgl1-mesa-glx`
- `libglib2.0-0`
- `ffmpeg`
- `wget`
- `git`

## 🤝 기여하기

1. 레포지토리를 포크하세요
2. 기능 브랜치를 생성하세요: `git checkout -b feature-name`
3. 변경사항을 커밋하세요: `git commit -am 'Add feature'`
4. 브랜치에 푸시하세요: `git push origin feature-name`
5. Pull Request를 제출하세요

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 라이선스가 부여됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- Meituan의 [YOLOv6](https://github.com/meituan/YOLOv6)
- 최적화된 추론을 위한 [ONNX Runtime](https://onnxruntime.ai/)
- 샘플 이미지를 위한 [COCO Dataset](https://cocodataset.org/)

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. [Issues](https://github.com/jisunclaralee/yolov6-to-onnx-test/issues) 페이지를 확인하세요
2. 상세한 설명과 함께 새로운 이슈를 생성하세요
3. 시스템 정보 및 오류 로그를 포함하세요

---

**컴퓨터 비전 커뮤니티를 위해 ❤️로 제작되었습니다**