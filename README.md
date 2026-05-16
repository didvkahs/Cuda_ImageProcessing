#Cuda_ImageProcessing

---

## 1. 개요 (Overview)

본 프로젝트는 CUDA(Compute Unified Device Architecture)를 사용하여 이미지의 명암 대비(Contrast)를 자동으로 개선하는 **히스토그램 평활화(Histogram Equalization)** 알고리즘을 구현한 것입니다. 

## 2. 구현 상세

### **Histogram 구성 (병렬 빈도수 추출) >**

- Histogram 배열
    - 픽셀의 밝기값을 고려해 0 ~ 255 사이의 값들을 담을 수 있는 배열을 만들었습니다. 이렇게 설정한 이유는 밝기값을 index로 사용하기 위해서입니다.
- race condition
    - 병렬처리의 경우 동일한 메모리 값에 데이터를 작성할 경우 발생합니다.
    - 이로 인해 잘못된 값들이 들어가거나, 누적이 안되는 문제가 발생합니다.
    - 이를 해결하기 위해 atomicAdd를 사용해 해결했습니다.

### **Equalization 구현 (순위 기반 재매핑) >**

- CDF
    - 이 함수를 사용하지 않았을 경우 어두운 이미지는 0이 되버리는 문제가 발생합니다. 이를 해결하기 위해 CDF를 사용했습니다.
    - 누적합을 사용할 경우, 특정 픽셀의 값들을 평균분포에서 어느 위치에 해당하는지 확인할 수 있습니다. 이를 이용해 픽셀의 밝기값을 변경합니다.
- Histogram
    - host에서 histogram의 값들을 equalized된 값들로 변경해, 이미지 데이터에 적용했습니다.

### Fuzzy bi Histogram equalization >

- 2025년 ICIVP에서 발표된 **"Fuzzy Bi-Histogram Equalization for Enhancement of Images"** 논문의 알고리즘을 구현하고, 기존 Histogram Equalization(HE)의 한계를 개선한 이미지 향상 기법을 다룹니다.
- 이미지의 모호성을 처리하기 위해 이미지의 Histogram을 Fuzzy화 하여 사용합니다.
- 가중 평균 기반 분할을 합니다. 이미지의 평균 밝기를 보존하기 위해 데이터의 무게중심을 구해 127이 아닌 이미지에 맞는 기준을 찾습니다.
- FBHE 기법을 사용한 결과 논문에 나온대로 기존 HE의 한계이던 과도한 밝기 변화나 노이즈 현상이 현저히 줄어들었습니다.

### IsoDataClustered Fuzzy histogram equalization >

- 앞선 Fuzzy bi Histgram equalization의 경우 Histogram을 단순 2개의 덩어리로 나누어 이미지 혁신적으로 개선했습니다. 이에 ISO DATA clustering을 사용하여 더 정교하게 Histogram을 나눈다면 더 확실한 개선이 일어날 것이라고 판단했습니다.
- 출력 결과 기존의 color map과 크게 다르지 않은 이미지가 출력됐습니다.
- 실패 원인
    - ISODATA clustering 은 이미지 내에서 비슷한 밝기를 가진 픽셀들을 찾아 분류하거나 양자화하는 것이 목적이기 때문입니다. 따라서 pl을 사용하여 fuzzy histogram을 여러개의 pl로 나눈다고 해도 iso data clustering을 사용한 것과 다르지 않습니다.
    - 또한, histogram equalization은 밝기의 분포를 넓히는 것이 목적이지만, cluster의 개수가 늘어날수록 분포를 넓게 펼치는 효과가 상쇄됩니다.
    - 각 클러스터마다의 경계에서 불연속성이 발생하여, 자연스러운 대비 향상이 아닌 color map과 같은 계단 현상이 발생합니다.

## 3. 참고 문헌

Hafijur Rahman and Tetsuya Shimamura, "Fuzzy Bi-Histogram Equalization for Enhancement of Images," *2025 International Conference on Image and Video Processing (ICIVP)*.
