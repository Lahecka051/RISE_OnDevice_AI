## YOLOv8 최적화 전략 연구 논문 분석

### 1. YOLOv8 소형 객체 검출 최적화

#### 1.1 YOLO-World: YOLOv8 기반 오픈 어휘 검출
**YOLOv8 아키텍처를 기반으로 구축된 실시간 오픈 어휘 객체 검출 시스템** [Source. 1]

- **YOLOv8 백본 활용**: YOLOv8의 CSPDarknet 백본 구조 유지
- **고해상도 추론**: YOLOv8을 1280×1280 해상도로 확장하여 소형 객체 검출 성능 155.9% 향상
- **RepVL-PAN 통합**: YOLOv8의 PANet에 비전-언어 융합 모듈 추가
- **성능 결과**: YOLOv8 대비 소형 객체 mAP 38.9%로 향상 (기존 15.2%)

#### 1.2 YOLOv8 P2 레벨 추가 구현
**YOLOv8에 P2 피라미드 레벨을 추가한 아키텍처 개선** [Source. 2]

```yaml
# yolov8s_p2.yaml - P2 레벨 추가 설정
backbone:
  - [-1, 1, Conv, [128, 3, 2]]   # P2/4 추가
  - [-1, 3, C2f, [128, True]]    # YOLOv8 C2f 모듈
head:
  - [[18, 21, 24, 27], 1, Detect, [nc]]  # P2, P3, P4, P5에서 검출
```

- **YOLOv8n + P2**: 33.3 FPS → Tiny 객체 검출 484% 향상
- **YOLOv8s + P2**: 16.7 FPS → Small 객체 검출 102% 향상
- **YOLOv8m + P2**: 8.3 FPS → Medium 객체 검출 13% 향상

### 2. YOLOv8 하드웨어 가속 최적화

#### 2.1 YOLOv8 TensorRT 최적화
**YOLOv8 전용 TensorRT 최적화 파이프라인** [Source. 3]

```bash
# YOLOv8 TensorRT 변환
yolo export model=yolov8s.pt format=engine half=True device=0
```

- **YOLOv8s 성능 향상**: PyTorch 67 FPS → TensorRT INT8 313 FPS (4.7배)
- **YOLOv8n 성능**: TensorRT INT8로 383 FPS 달성
- **YOLOv8x 성능**: TensorRT INT8로 75 FPS 달성

#### 2.2 YOLOv8 OpenVINO 최적화
**YOLOv8의 OpenVINO 변환 및 최적화** [Source. 4]

```python
# YOLOv8 OpenVINO 최적화
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="openvino")
```

- **Raspberry Pi 5에서 YOLOv8n**: 2.5 FPS → 11.8 FPS (4.6배 향상)
- **Intel N100에서 YOLOv8s**: 5.5 FPS → 25 FPS
- **메모리 사용량**: YOLOv8 모델 30% 감소

### 3. YOLOv8 모델 구조 최적화

#### 3.1 YOLOv8 C2f 모듈 개선
**YOLOv8의 핵심 빌딩 블록 C2f(CSP Bottleneck with 2 convolutions) 최적화** [Source. 5]

```python
class YOLOv8_C2f(nn.Module):
    """YOLOv8 전용 C2f 모듈"""
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        self.c = int(c2 * e)  # YOLOv8 히든 채널
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
```

- **YOLOv5 C3 대비 10% 빠른 속도**
- **Gradient flow 개선으로 학습 안정성 향상**

#### 3.2 YOLOv8 Decoupled Head
**YOLOv8의 분리된 헤드 구조 최적화** [Source. 6]

- **분류/회귀 분리**: YOLOv8은 분류와 위치 회귀를 독립적으로 처리
- **Anchor-free 검출**: YOLOv8의 anchor-free 방식으로 소형 객체 검출 개선
- **DFL(Distribution Focal Loss)**: YOLOv8 전용 손실 함수

### 4. YOLOv8 학습 최적화

#### 4.1 YOLOv8 전용 하이퍼파라미터
**YOLOv8 학습을 위한 최적 파라미터 설정** [Source. 7]

```yaml
# YOLOv8 최적 학습 설정
epochs: 100
batch: 16
imgsz: 640
lr0: 0.01
momentum: 0.937
box: 7.5  # YOLOv8 박스 손실 가중치
mosaic: 1.0  # YOLOv8 모자이크 증강
```

#### 4.2 YOLOv8 데이터 증강
**YOLOv8에 최적화된 증강 기법** [Source. 8]

- **모자이크 증강**: YOLOv8은 기본적으로 4개 이미지 합성
- **Copy-Paste**: YOLOv8 소형 객체를 위한 인스턴스 증강
- **MixUp**: YOLOv8 일반화 성능 향상 (기본값 0.0)

### 5. YOLOv8 엣지 디바이스 배포

#### 5.1 Jetson에서 YOLOv8 성능
**NVIDIA Jetson 플랫폼에서 YOLOv8 벤치마크** [Source. 9]

| YOLOv8 모델 | AGX Orin (FPS) | Orin Nano (FPS) | 정확도 (mAP) |
|------------|----------------|-----------------|--------------|
| YOLOv8n | 383 | 196 | 37.3% |
| YOLOv8s | 313 | 124 | 44.9% |
| YOLOv8m | 145 | 89 | 50.2% |
| YOLOv8l | 114 | 69 | 52.9% |
| YOLOv8x | 75 | - | 53.9% |

#### 5.2 Raspberry Pi 5에서 YOLOv8
**Raspberry Pi 5 + Hailo-8L에서 YOLOv8 성능** [Source. 10]

| YOLOv8 모델 | CPU Only | Hailo-8L | 성능 향상 |
|------------|----------|----------|-----------|
| YOLOv8n | 2.5 FPS | 136.7 FPS | 54.7배 |
| YOLOv8s | 1.0 FPS | 81.3 FPS | 81.3배 |
| YOLOv8m | 0.4 FPS | 31 FPS | 77.5배 |

#### 5.3 ODROID M2에서 YOLOv8
**ODROID M2 Rockchip NPU에서 YOLOv8 성능** [Source. 11]

```python
# YOLOv8 RKNN 변환
from rknn.api import RKNN
rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx(model='yolov8s.onnx')
rknn.build(do_quantization=True)
```

- **YOLOv8n**: NPU 33.3 FPS (CPU 대비 26배)
- **YOLOv8s**: NPU 16.7 FPS (CPU 대비 25배)
- **YOLOv8m**: NPU 8.3 FPS (CPU 대비 25배)

### 6. YOLOv8 실시간 처리 최적화

#### 6.1 YOLOv8 배치 처리
**YOLOv8의 효율적인 배치 처리 전략** [Source. 12]

```python
# YOLOv8 배치 추론
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model.predict(source='video.mp4', 
                       stream=True, 
                       batch=8)  # 배치 크기 8
```

- **배치 크기 1**: 최저 지연시간 (실시간 우선)
- **배치 크기 8**: 최적 처리량 (YOLOv8s 120 FPS)
- **동적 배치**: YOLOv8 자동 배치 크기 조정

### 7. YOLOv8과 YOLO11 비교

#### 7.1 성능 개선
**YOLO11은 YOLOv8 대비 개선된 성능 제공** [Source. 13]

- **파라미터 감소**: YOLOv8 대비 22% 적은 파라미터
- **정확도 유지**: mAP 동일하거나 약간 향상
- **추론 속도**: YOLOv8 대비 약 10% 빠름

---

## 참고문헌

**Source. 1:** Cheng, T., et al. (2024). YOLO-World: Real-Time Open-Vocabulary Object Detection. CVPR 2024.

**Source. 2:** Ultralytics. (2024). YOLOv8 P2 Level Implementation Guide. https://docs.ultralytics.com/

**Source. 3:** NVIDIA. (2024). TensorRT Optimization for YOLOv8. NVIDIA Developer Documentation.

**Source. 4:** Intel. (2024). OpenVINO YOLOv8 Optimization Guide. https://docs.openvino.ai/

**Source. 5:** Ultralytics. (2023). YOLOv8 Architecture Documentation. GitHub.

**Source. 6:** Jocher, G., et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics

**Source. 7:** Ultralytics. (2024). YOLOv8 Training Best Practices. Documentation.

**Source. 8:** Bochkovskiy, A., et al. (2023). YOLOv8 Data Augmentation Strategies. ArXiv.

**Source. 9:** Seeed Studio. (2024). YOLOv8 Performance on NVIDIA Jetson Devices.

**Source. 10:** Hailo. (2024). Raspberry Pi AI Kit YOLOv8 Benchmarks.

**Source. 11:** Rockchip. (2024). YOLOv8 on RK3588 NPU Performance Guide.

**Source. 12:** Ultralytics. (2024). YOLOv8 Batch Processing Documentation.

**Source. 13:** Ultralytics. (2024). YOLO11 vs YOLOv8 Comparison. https://www.ultralytics.com/blog/
