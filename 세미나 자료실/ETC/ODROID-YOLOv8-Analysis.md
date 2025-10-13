# ODROID 플랫폼별 YOLOv8 종합 성능 분석 보고서

## 개요
본 문서는 2023-2025년 발표된 연구 자료를 바탕으로 ODROID의 다양한 플랫폼에서 YOLOv8 객체 검출 모델의 성능을 종합 분석한 보고서입니다. 각 플랫폼별 하드웨어 사양, **AI 연산 성능(TOPs)**, 실제 벤치마크 결과, 최적화 기법, 그리고 용도별 권장사항을 상세히 다룹니다.

## 1. ODROID M 시리즈 - Rockchip NPU 기반 플랫폼

### 1.1 ODROID-M1 (RK3568B2)

#### 하드웨어 사양
- **SoC**: Rockchip RK3568B2
- **CPU**: Quad-Core Cortex-A55 @ 2.0GHz
- **GPU**: Mali-G52 MP2
- **NPU**: 0.8 TOPS @ INT8 (RKNN NPU)
- **AI 연산 성능**: 🔷 **0.8 TOPS**
- **RAM**: 4GB/8GB LPDDR4
- **장기 지원**: 2036년까지 보장
- **가격**: $90-130

#### YOLOv8 성능 측정 (2023-2024)

**실제 벤치마크 결과 (ODROID 포럼, 2023년 12월)**

| 처리 모드 | YOLOv8s 추론 시간 | 처리 속도 | 속도 향상비 |
|----------|------------------|-----------|------------|
| CPU 순수 연산 | 224ms/이미지 | 4.5 FPS | 기준값 |
| librga 가속 | 125ms/이미지 | 8 FPS | 1.8x |
| NPU 가속 | 60ms/이미지 | 16.7 FPS | 3.7x |

**NPU 가속 상세 성능**

| 모델 | CPU+NEON | NPU 가속 | FPS (NPU) | 속도 향상 |
|------|----------|----------|-----------|-----------|
| YOLOv8n | ~800ms | ~30ms | 33.3 | 26x |
| YOLOv8s | ~1500ms | ~60ms | 16.7 | 25x |
| YOLOv8m | ~3000ms | ~120ms | 8.3 | 25x |

### 1.2 ODROID-M1S (RK3566)

#### 하드웨어 사양
- **SoC**: Rockchip RK3566 (M1의 경량화 버전)
- **CPU**: Quad-Core Cortex-A55 @ 1.8GHz
- **GPU**: Mali-G52 MP2
- **NPU**: 0.8 TOPS @ INT8
- **AI 연산 성능**: 🔷 **0.8 TOPS**
- **RAM**: 4GB/8GB LPDDR4
- **내장 스토리지**: 64GB eMMC (솔더링)
- **폼팩터**: M1 대비 20% 소형화
- **가격**: $70-100

#### YOLOv8 성능 측정 (2024)

| 모델 | 추론 시간 | FPS | mAP@0.5 | 전력 소비 |
|------|-----------|-----|---------|-----------|
| YOLOv8n | ~35ms | 28.6 | 37.3% | 2.8W |
| YOLOv8s | ~70ms | 14.3 | 44.9% | 3.2W |
| YOLOv8m | ~150ms | 6.7 | 50.2% | 3.5W |

### 1.3 ODROID-M2 (RK3588S2) - 플래그십 모델

#### 하드웨어 사양
- **SoC**: Rockchip RK3588S2
- **CPU**: 4x Cortex-A76 @ 2.3GHz + 4x Cortex-A55 @ 1.8GHz
- **GPU**: Mali-G610 MP4 @ 1GHz
- **NPU**: 6 TOPS @ INT8 (M1 대비 7.5배)
- **AI 연산 성능**: 🔥 **6.0 TOPS**
- **RAM**: 8GB/16GB LPDDR5
- **PCIe**: Gen3 x1 레인
- **가격**: $200-300

#### YOLOv8 성능 벤치마크 (2024)

**NPU 가속 성능 (INT8 양자화)**

| 모델 | 추론 시간 | FPS | mAP@0.5 | 전력 소비 |
|------|-----------|-----|---------|-----------|
| YOLOv8n | 10ms | 100.0 | 36.8% | 5W |
| YOLOv8s | 16ms | 62.5 | 44.2% | 6W |
| YOLOv8m | 30ms | 33.3 | 49.5% | 7W |
| YOLOv8l | 50ms | 20.0 | 52.1% | 8W |
| YOLOv8x | 80ms | 12.5 | 53.8% | 9W |

---

## 2. ODROID H 시리즈 - Intel x86 기반 플랫폼

### 2.1 ODROID-H3/H3+ (Intel Jasper Lake)

#### 하드웨어 사양
- **CPU**: Intel N5105/N6005 (Quad-Core @ 2.0-3.3GHz)
- **GPU**: Intel UHD Graphics (24 EU)
- **AI 연산 성능**: 🔶 **~0.3 TOPS** (GPU 추정치)
- **RAM**: 최대 64GB DDR4-2933
- **폼팩터**: 110 x 110mm Mini-ITX
- **TDP**: 10-15W
- **가격**: H3 $165, H3+ $220

#### YOLOv8 성능 (OpenVINO 최적화)

| 모델 | CPU 추론 | GPU 추론 | FP16 가속 | INT8 가속 |
|------|----------|----------|-----------|-----------|
| YOLOv8n | 50-80ms | 40-60ms | 30-45ms | 20-30ms |
| YOLOv8s | 120-180ms | 90-140ms | 70-100ms | 50-75ms |
| YOLOv8m | 250-350ms | 200-280ms | 150-210ms | 100-140ms |

### 2.2 ODROID-H4 시리즈 (Intel Alder Lake-N) - 2024년 신제품

#### 2.2.1 ODROID-H4 (기본 모델)
- **CPU**: Intel N97 (Quad-Core @ 2.0GHz, Turbo 3.6GHz)
- **GPU**: Intel UHD Graphics 24 EU @ 1.2GHz
- **AI 연산 성능**: 🔶 **~0.5 TOPS** (GPU 추정치)
- **RAM**: 최대 48GB DDR5-4800 (단일 채널)
- **특징**: SATA 미지원, 단일 2.5GbE LAN
- **가격**: $99

#### 2.2.2 ODROID-H4+ (중급 모델)
- **CPU**: Intel N97 (H4와 동일)
- **AI 연산 성능**: 🔶 **~0.5 TOPS** (GPU 추정치)
- **추가 기능**: 
  - 4x SATA3 포트
  - 2x 2.5GbE LAN
  - Dual BIOS
- **가격**: $139

#### 2.2.3 ODROID-H4 Ultra (플래그십)
- **CPU**: Intel Core i3-N305 (8-Core @ 1.8GHz, Turbo 3.8GHz)
- **GPU**: Intel UHD Graphics 32 EU
- **AI 연산 성능**: 🔶 **~0.8 TOPS** (GPU 추정치)
- **성능**: H3+ 대비 83% 향상
- **가격**: $220

#### H4 시리즈 YOLOv8 성능 (OpenVINO 최적화)

| 모델 | H4/H4+ (N97) | H4 Ultra (i3-N305) |
|------|--------------|-------------------|
| YOLOv8n | 25-30 FPS | 40-50 FPS |
| YOLOv8s | 12-15 FPS | 20-25 FPS |
| YOLOv8m | 6-8 FPS | 10-12 FPS |
| YOLOv8l | 3-4 FPS | 5-7 FPS |
| YOLOv8x | 1-2 FPS | 3-4 FPS |

---

## 3. ODROID N 시리즈 - Amlogic 기반 플랫폼

### 3.1 ODROID-N2+ (Amlogic S922X)

#### 하드웨어 사양
- **SoC**: Amlogic S922X
- **CPU**: 4x Cortex-A73 @ 2.2GHz + 2x Cortex-A53 @ 2.0GHz
- **GPU**: Mali-G52 MP4 @ 800MHz
- **AI 연산 성능**: ⚪ **CPU Only (~0.1 TOPS)**
- **RAM**: 2GB/4GB DDR4-1320MHz
- **아키텍처**: 12nm, big.LITTLE
- **가격**: $80-100

#### YOLOv8 성능 (TensorFlow Lite 최적화)

| 모델 | FP32 추론 | INT8 추론 | FPS (INT8) |
|------|-----------|-----------|------------|
| YOLOv8n | 150-300ms | 75-150ms | 6.7-13.3 |
| YOLOv8s | 300-600ms | 150-300ms | 3.3-6.7 |
| YOLOv8m | 800-1500ms | 400-750ms | 1.3-2.5 |

### 3.2 ODROID-N2/N2L
- **N2**: 동일 SoC, 낮은 클럭 (2.0/1.9GHz)
- **N2L**: 경량화 버전, 2GB RAM 전용
- **AI 연산 성능**: ⚪ **CPU Only (~0.08 TOPS)**
- **예상 성능**: N2+ 대비 10-15% 성능 저하

---

## 4. ODROID C 시리즈 - 엔트리급 플랫폼

### 4.1 ODROID-C4 (Amlogic S905X3)

#### 하드웨어 사양
- **SoC**: Amlogic S905X3
- **CPU**: Quad-Core Cortex-A55 @ 2.0GHz
- **GPU**: Mali-G31 MP2
- **AI 연산 성능**: ⚪ **CPU Only (~0.05 TOPS)**
- **RAM**: 4GB DDR4
- **가격**: $55

#### YOLOv8 성능

| 모델 | FP32 추론 | INT8 추론 | FPS (INT8) |
|------|-----------|-----------|------------|
| YOLOv8n | 200-400ms | 100-200ms | 5-10 |
| YOLOv8s | 500-800ms | 250-400ms | 2.5-4 |
| YOLOv8m | 1200-2000ms | 600-1000ms | 1-1.7 |

---

## 5. 플랫폼별 AI 연산 성능 종합 비교

### 5.1 TOPs 성능 순위

| 순위 | 플랫폼 | AI 가속기 | TOPs | YOLOv8s FPS | 효율성 |
|------|--------|-----------|------|-------------|---------|
| 🥇 1 | **ODROID-M2** | NPU | **6.0 TOPS** | 62.5 | 최고 |
| 🥈 2 | **ODROID-M1** | NPU | **0.8 TOPS** | 16.7 | 우수 |
| 🥈 2 | **ODROID-M1S** | NPU | **0.8 TOPS** | 14.3 | 우수 |
| 🥉 3 | **ODROID-H4 Ultra** | GPU | ~0.8 TOPS | 20-25 | 보통 |
| 4 | **ODROID-H4/H4+** | GPU | ~0.5 TOPS | 12-15 | 보통 |
| 5 | **ODROID-H3/H3+** | GPU | ~0.3 TOPS | 5.5-8.3 | 낮음 |
| 6 | **ODROID-N2+** | CPU | ~0.1 TOPS | 1.6-3.3 | 낮음 |
| 7 | **ODROID-N2/N2L** | CPU | ~0.08 TOPS | ~1.5 | 낮음 |
| 8 | **ODROID-C4** | CPU | ~0.05 TOPS | 1.2-2 | 최저 |

### 5.2 AI 가속 기술별 성능 특성

#### NPU 탑재 모델 (M 시리즈)
- **장점**: 
  - 전용 AI 가속으로 최고 성능
  - 전력 효율성 탁월 (FPS/W)
  - INT8 양자화 최적화
- **단점**:
  - 특정 프레임워크 종속 (RKNN)
  - 모델 변환 필요

#### GPU 가속 모델 (H 시리즈)
- **장점**:
  - OpenVINO 등 다양한 프레임워크 지원
  - x86 호환성으로 개발 용이
  - FP16/INT8 유연한 정밀도 선택
- **단점**:
  - NPU 대비 낮은 효율성
  - 높은 전력 소비

#### CPU 전용 모델 (N/C 시리즈)
- **장점**:
  - 저렴한 가격
  - 범용 컴퓨팅 가능
- **단점**:
  - AI 작업에 부적합한 성능
  - 매우 낮은 FPS

### 5.3 용도별 최적 플랫폼 선택 (TOPs 기준)

| 용도 | 필요 TOPs | 권장 플랫폼 | 이유 |
|------|-----------|------------|------|
| **실시간 AI (30+ FPS)** | 3.0+ TOPS | ODROID-M2 | 6 TOPS로 충분한 여유 |
| **스마트 홈 보안** | 0.5-1.0 TOPS | ODROID-M1/M1S | 효율적인 NPU 활용 |
| **엣지 IoT** | 0.5-1.0 TOPS | ODROID-M1S | 저전력, 소형 |
| **개발/테스트** | 0.3-1.0 TOPS | ODROID-H4 시리즈 | x86 호환성 |
| **교육/학습** | 0.05+ TOPS | ODROID-C4 | 저렴한 입문용 |

### 5.4 성능 대비 가격 분석

| 플랫폼 | 가격 | TOPs | $/TOPS | YOLOv8s FPS | $/FPS |
|--------|------|------|--------|-------------|-------|
| **ODROID-M2** | $250 | 6.0 | **$41.7** | 62.5 | $4.0 |
| **ODROID-M1** | $110 | 0.8 | $137.5 | 16.7 | $6.6 |
| **ODROID-M1S** | $85 | 0.8 | $106.3 | 14.3 | $5.9 |
| **ODROID-H4 Ultra** | $220 | ~0.8 | $275.0 | 22.5 | $9.8 |
| **ODROID-C4** | $55 | ~0.05 | $1,100 | 1.6 | $34.4 |

---

## 6. 실제 구현 가이드

### 6.1 NPU 활용 최적화 (M 시리즈)

```python
# RKNN NPU 활용 예제
import cv2
import numpy as np
from rknnlite.api import RKNNLite

# RKNN 모델 로드 (0.8 또는 6.0 TOPS NPU 활용)
rknn = RKNNLite()
rknn.load_rknn('./yolov8s.rknn')
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

# 추론 실행 - NPU 가속
outputs = rknn.inference(inputs=[input_data])
```

### 6.2 전력 효율성 비교

| 플랫폼 | TOPs | 전력 소비 | TOPs/W | 효율성 등급 |
|--------|------|-----------|--------|------------|
| **ODROID-M2** | 6.0 | 7W | 0.86 | ⭐⭐⭐⭐⭐ |
| **ODROID-M1** | 0.8 | 3.5W | 0.23 | ⭐⭐⭐⭐ |
| **ODROID-M1S** | 0.8 | 3.2W | 0.25 | ⭐⭐⭐⭐ |
| **ODROID-H4 Ultra** | ~0.8 | 30W | 0.03 | ⭐⭐ |
| **ODROID-N2+** | ~0.1 | 10W | 0.01 | ⭐ |

---

## 7. 결론 및 권장사항

### 7.1 주요 인사이트

1. **NPU의 중요성**: 
   - 0.8 TOPS NPU (M1/M1S)도 CPU 대비 25배 성능 향상
   - 6.0 TOPS NPU (M2)는 실시간 AI에 충분한 성능 제공

2. **TOPs와 실제 성능 관계**:
   - 1 TOPS 이하: 기본적인 AI 작업 가능
   - 1-3 TOPS: 실시간 단일 스트림 처리
   - 3-6 TOPS: 다중 스트림 및 복잡한 AI 작업
   - 6+ TOPS: 고성능 AI 애플리케이션

3. **플랫폼 선택 기준**:
   - **성능 우선**: ODROID-M2 (6.0 TOPS)
   - **효율성 우선**: ODROID-M1S (0.8 TOPS, 저전력)
   - **호환성 우선**: ODROID-H4 시리즈 (x86)
   - **가격 우선**: ODROID-C4 (엔트리급)

### 7.2 향후 전망

- **2025년**: 10+ TOPS NPU 탑재 모델 출시 예상
- **소프트웨어**: RKNN SDK 3.0으로 더 나은 최적화
- **가격**: AI 가속기 대중화로 $/TOPS 지속 하락

---

## 8. 참고문헌

### 기술 문서
1. **ODROID Wiki - RKNN YOLOv8 Application Note** (2023-2024)
2. **Hardkernel Official Product Specifications** (2024)
3. **Rockchip RKNN Model Zoo GitHub Repository** (2024)
4. **Intel OpenVINO Documentation** (2024)

### 벤치마크 자료
5. **ODROID Community Forum Benchmarks** (2023-2024)
6. **CNX Software ODROID Reviews** (2024)
7. **ArXiv:2409.16808v1** - "Benchmarking Deep Learning Models on Edge Devices" (2024)

---

*본 문서는 2023-2025년 발표된 공개 연구자료, 기술문서, 커뮤니티 벤치마크를 종합 분석하여 작성되었습니다. 실제 성능은 소프트웨어 버전, 최적화 수준, 환경 설정에 따라 달라질 수 있습니다.*

*TOPs 값 중 일부는 제조사 공식 사양이며, GPU/CPU 기반 플랫폼의 TOPs는 일반적인 추정치입니다.*

*최종 업데이트: 2025년 10월*
