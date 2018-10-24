
# Lecture 5 note

## Convolutional Neural Network

Convolutional Layer : 공간적 구조를 유지

### Neural Network의 역사

- Mark 1 Perceptron machine
Wx + b와 유사하지만 출력값이 1 또는 0 뿐.    
Update Rule도 존재. Back Prop과 비슷하다.

- Adaline / Madaline
최초의 Multilayer Perceptron Network    
Back Prop과 같은 학습 알고리즘은 없음.

- Rumelhart가 Back Prop 제안 in 1986

- DNN의 학습 가능성 2006

### CNN의 역사

- 뉴런이 oriented edge와 shape에 반응 한다는 것을 알아냄.

- Topographical mapping in the cortex : 근처에 존재하는 뉴런은 visual field 에서도 근처를 담당하는 것을 알아냄.

- Hierarchical organization : 뉴런이 계층 구조를 가진다는 것을 알아냄.

- Neurocognitron : simple cell(학습 가능한 파라미터가 있음)과 complex cell(pooling 같은 것으로 구성, 작은 변화에 좀 더 강인함)을 교차. 

- NN을 학습시키기 위해 Back Prop과 gradient-based learning을 적용

- AlexNet : 신경망이 더 깊어지고 커졌다. 대규모 데이터를 활용할 수 있었고, GPU가 파워풀 해졌다.

### CNN의 사용처

- 이미지 검색
- Object Detection (Localization)
- 자율주행
- 얼굴인식
- Pose Recognition
- 게임 학습
- 의학 진단, 천문 이미지, 표지판 인식, 고래 분류, 항공 지도를 가지고 길 표시
- Image Captioning
- 예술 작품 생성, 이미지를 특정 화풍으로 재생성

### CNN 동작 방식

#### Fully Connected Layer

x : 32 x 32 x 3 image => 3072 x 1 vector => input
W : 10 x 3072 vector
Wx : 10 x 1

#### Convolution Layer

- 기존의 이미지 구조를 보존
- 작은 필터가 W의 역할을 함. 이미지를 슬라이딩 하면서 부분적으로 Dot Product를 함.
- Conv 연산을 하고 나온 값을 Output Activation Map에 저장함.
- 여러 개의 필터를 사용. 따라서 여러개의 Activation Map이 생성됨.
- 이런 레이어를 여러 개를 쌓아서 NN을 만든다.

## Spatial Dimentsion

### stride

Output Size = (N - F) / stride + 1

### Zero pad

하는 이유 : 레이어를 거치면서도 같은 사이즈를 유지하기 위해
Output Size = (N + 2Z - F) / stride + 1

### Parameter 수

필터당 파라미터 수 : F * F * D + 1
총 파라미터 수  : (필터당 파라미터 수) * (필터 수)