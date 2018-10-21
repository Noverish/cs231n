
# Lecture 9 note

## LeNet

LeNet은 성공적으로 산업에 적용된 최초의 ConvNet이다.

## AlexNet

2012년에 나옴. 최초의 Large scale CNN.    
conv - pool - normalization 구조가 두 번 반복된다. 그리고 conv가 3개가 더 붙고 max pooling layer가 붙고 FC가 3개 붙는다.    
ImageNet으로 학습시키는 경우 입력의 크기가 227 x 227 x 3이다.

1\. Layer 1 (CONV1)    
필터 : 11 x 11, stride=4, 96개    
총 파라미터 수 : 11 x 11 x 3 x 96    
출력의 크기 : (227-11)/4+1 = 55 => 55 x 55 x 96

2\. Layer 2 (POOL1)    
필터 : 3 x 3, stride=2    
총 파라미터 수 : 0    
출력의 크기 : (55-3)/2 + 1 = 27 => 27 x 27 x 96

3\. 마지막 Layer (FC8 이후)    
sofxmax 레이어이다.

- ReLU를 처음 사용한 모델
- Local Response Normalization Layer는 채널간의 Normalization을 위한 것인데 요즘은 잘 사용하지 않는다. 큰 효과가 없는 것으로 알려졌기 때문.
- 엄청난 Data Augmentation 
- Dropout 0.5
- batch size = 128
- SGD Momentum = 0.9
- LR = 1e-2 -> 1e-10 (val acc가 줄어들지 않을 때 LR을 줄임)
- L2 weight decay = 5e-4
- 7 CNN ensemble

모델이 두 개로 나눠져서 서로 교차한다. 당시에는 GPU 메모리가 3GB 밖에 안해서 두 GPU에 나누어서 학습했기 때문. 그래서 CONV1의 출력을 잘 살펴보면 depth가 96이 아니라 48이다.    
Conv1, 2, 4, 5는 48개의 feature map만 사용    
Conv 3, FC 6, 7, 8은 모든 Feature map을 사용    
논문에는 224 x 224로 되어 있지만 실제는 227 x 227    
Image Classification Benchmark의 2012년도 우승 모델    
최초의 CNN기반 우승 모델

Q. 왜 AlexNet이 기존보다 더 뛰어난가?    
A. 딥러닝과 ConvNet의 힘이다. 기존과는 완전히 다른 방법이다.

## ZFNet

2013년도 우승 모델은 ZFNet. AlexNet의 hyperparameter를 조절하여 더 높은 성능을 냄.

## VGGNet

2014년부터 신경망이 깊어진다. 16 ~ 19개의 레이어를 가짐    
더 작은 필터 사용. 항상 3x3 필터만 사용. 주기적으로 Pooling을 수행.    
작은 필터를 사용하는 이유 => 파라미터가 적어지기 때문, 그래서 레이어를 더 많이 쌓을 수 있다. depth를 더 깊게 할 수 있다.

Receptive Field : 필터가 한 번에 볼 수 있는 입력의 Spacial area    
3개의 3x3 필터(stride 1)는 7x7 필터와 같은 effective receptive field를 가진다. 피라미드 처럼 그려보면 그렇게 된다.
이 두 필터의 파라미터 수를 세보면 3개의 3x3 필터는 3 * (3^2 * C^2) 이고 1개의 7x7 필터는 (7^7 * C^2)이다.

총 메모리 사용량은 입력 한 개당 100MB, backward pass까지 고려하면 더 많은 메모리가 필요함. 그래서 VGGNet은 메모리 사용량이 많은 편.

총 파라미터는 138M, AlexNet은 60M.

보통 신경망의 깊이(depth)는 학습 가능한 파라미터가 있는 레이어의 수 (Conv, FC)

Q. 왜 하나의 Conv Layer에 여러 개의 필터가 존재하느냐?    
A. 하나의 필터는 하나의 feature map을 만든다. 각 feature map은 서로 다른 패턴을 인식한다.

Q. 신경망이 깊어질 수록 레이어의 필터 개수를 늘려야 하나?    
A. 니 맘이다. 일반적으로는 늘린다. 신경망이 깊어질수록 feature map의 사이즈는 작아지는데 계산량을 일정하게 유지시키기 위해서 필터 개수를 늘린다.

Q. Softmax Loss 대신 SVM Loss를 사용해야 되는가?    
A. 니 맘이다. 일반적으로는 Softmax를 사용

Q. forward pass에서 전에 계산 했던 값을 버려도 되지 않는가?    
A. backward pass에서 다 쓰니까 버리지 말라

잘 보면 신경망 초반에는 메모리를 많이 쓰고 파라미터 수가 적고 신경망 후반에는 메모리를 적게 쓰고 파라미터 수가 많다. 그래서 최근 신경망 중에 파라미터 수를 줄이기 위해 FC를 아예 없애버리는 경우도 있다.

- VGGNet은 Classification에서 2등, Localization에서 1등
- AlexNet과 비슷한 학습 과정을 거친다.
- AlexNet에 있는 Local Response Normalization Layer가 없다. 도움이 안 되기 때문
- VGG19가 VGG16보다 더 많은 메모리를 쓰고 성능이 약간 더 좋다. 하지만 일반적으로 VGG16을 쓴다.
- Ensemble을 사용하여 성능 향상
- FC7 레이어는 데이터의 일반적으로 feature를 잘 나타내고 있는 것으로 나온다. 그래서 다른 작업을 할 때도 쓰인다.

## GoogLeNet

22개의 레이어를 가지고 있음. 높은 계산량을 효율적으로 하는 네트워크를 디자인함. Inception module을 사용함. FC 레이어가 없다. 전체 파라미터 수가 5M.

### Inception Module

배경 : 좋은 local network topology를 만들고 싶었고 network within a network라는 개념으로 local topology를 구현, 이를 쌓아 올림

이 local network를 inception module이라고 부름. 동일한 입력을 받는 서로 다른 다양한 필터가 병렬적으로 존재함. 이렇게 병렬적으로 필터를 거친 값들을 depth 방향으로 합친다.

이 방법의 문제는 바로 계산이 비싸다는 것이다.    
입력 28 * 28 * 256 이면    
128개의 1x1 필터를 거친 출력 : 28 * 28 * 128    
192개의 3x3 필터를 거친 출력 : 28 * 28 * 192    
96개의 5x5 필터를 거친 출력 : 28 * 28 * 96    
3x3 Max Pooling을 거친 출력 : 28 * 28 * 256    
총 out은 28 * 28 * 672가 된다. depth가 너무 늘어나버렸다.    
192개의 3x3 필터의 총 계산량 = 28 * 28 * 192 * 3 * 3 * 256    
이 모든 레이어의 계산량을 합치면 854M개가 된다. 근데 module을 거쳐갈 수록 depth가 점점 더 많아진다.

그래서 bottleneck layer를 만들었다. 1x1 필터를 가지고 depth를 줄인다.     
3x3, 5x5의 필터를 거치기 전에 1x1 필터를 거치고, Max Pooling을 한 후에 1x1 필터를 거치게 한다. 이러한 필터들을 bottleneck layer라고 한다.

bottleneck layer의 depth를 64로 한 뒤 계산량을 다시 세보면 358M로 줄어들게 된다.

Q. 1x1 conv layer가 정보 손실을 야기하지 않나?    
A. 그럴 수 있다. 하지만 불필요한 정보가 있는 feature들을 결합한다고 볼 수도 있다. 그냥 일반적으로 잘 되더라. 이걸 하는 것은 계산 복잡도를 줄이기 위함이다.

### 구조

GoogLeNet의 앞단(stem)에는 일반적인 네트워크 구조가 들어간다.    
중간에는 Inception Module이 쌓여있는데 조금씩 생긴게 다르다.    
마지막에는 classifier 결과를 출력.

### Auxiliary Classifier

신경망 중간에 줄기가 뻗어 있는데 이를 보조분류기(Auxiliary Classifier)라고 한다. 일반적인 네트워크 구조인데 여기서도 loss를 구한다.
왜 이렇게 하냐면 신경망이 너무 깊기 때문이다. 보조분류기를 통해 추가적인 gradient를 얻을 수 있고 중간 레이어의 학습에 도움이 된다.

Q. 보조 분류기에서 나온 결과를 최종 분류에 사용할 수 있나?    
A. 기억이 안 난다. 확인해 봐라.

Q. bottleneck을 만들 때 1x1 conv layer 말고 다른 걸로도 할 수 있나?    
A. 그래도 되는데 이거는 차원 축소도 있고 conv layer이기 때문에 backprop으로 학습할 수도 있다.

Q. 모든 Inception Module 레이어의 가중치가 같나?    
A. 아니다.

Q. 왜 보조 분류기를 통해 앞 단의 layer의 gradient를 전달하느냐?    
A. 네트워크가 깊기 때문에 맨 뒤에서 전달하다 보면 0이 되버린다.

### 특징

- 22개의 레이어
- Inception Module이 있다.
- FC가 없다.
- AlexNet보다 파라미터수가 12배 적다.

## ResNet

15년도 우승. 신경망의 깊이가 극적으로 늘어남. 152개의 레이어. 모든 대회를 모조리 우승.

### Residual Connections

일반 CNN을 엄청 깊게 쌓으면 성능이 좋아지는가? VGG에 conv-pool 레이어를 깊게 쌓으면 성능이 좋아지는가? 그건 아니다.
실제 테스트를 해보면 56 layer가 20 layer보다 Train, Test 성능이 안 좋다.
56 layer면 보통 overfit 하다고 예상하는데 이상하게 Train error도 56 layer가 더 높다.

ResNet 저자들은 이런 문제가 Optimization에서 생긴다고 본다.    
신경망이 깊어지면 성능이 안 좋아지는 것을 그렇다 쳐도 성능이 비슷하게는 나와야 하는 거 아닌가? 라고 생각했다.    
얕은 신경망의 가중치를 깊은 신경망에 복사 하고 남는 레이어는 input과 output이 같게 하면 그렇게 되는 건데.    
이 생각을 모델에서 사용하려고 해서 생긴게 Residual Connection 이다.

일반적인 레이어는 X -> H(X) 모양인데 이렇게 바로 H를 학습하는 것은 어렵다. X -> F(X) + X 처럼 변화량의 함수만 학습하는 것은 더 쉽다.    
윗 줄의 내용은 저자의 가정이다. 

한 Residual Block은 2개의 3x3 conv layer로 이루어져 있다. 이 구성은 경험적으로 알아낸 것이다.
이것을 높게 쌓아 올린다. ResNet은 150개 까지 쌓아 올릴 수 있다.
주기적으로 필터를 두 배씩 늘리고 stride2를 이용하여 downsampling을 수행한다.

### 구조

네트워크 초반에는 Conv Layer가 붙고 네트워크 후반에는 FC Layer가 없다.
그 대신 Global Average Pooling Layer를 사용한다. GAP는 하나의 Feature Map 전체를 Average Pooling 한다.

ImageNet 문제를 위해 152개 까지 쌓아 올렸다.
네트워크의 깊이가 50 이상일 때 Bottleneck Layer를 사용한다.
28\*28\*256의 입력값이 1x1,64 -> 3x3,64 -> 1x1,256를 통과하여 같은 크기로 나온다.

### 특징

- 모든 Conv Layer다음에 Batch Norm을 사용한다.
- 초기화에는 Xavier/2를 사용
- SGD + Momentum (0.9)
- Learing Rate : 0.1 validation error가 정체하면 10으로 나눈다.
- Mini-batch size 256
- Weight decay 1e-5
- No Dropout Used

### 실험 결과

- 매우 깊은 신경망에서도 잘 동작하고
- back prob에서 gradient flow를 잘 가져온다.
- 네트워크가 깊어질 수록 Training Error는 줄어든다. 늘어나는 경우는 없었다.

## 결론

\-        | 효율 | 메모리 | 계산량 | 성능 | forward 시간
----------|-|-|-|-|-
AlexNet   | 낮다 | 많다 | 적다 | 별로 | 
VGGNet    | 낮다 | 많다 | 많다 | 적당 | 가장 길다
GoogLeNet | 높다 | 적다 | 적다 | 적당 | 
ResNet    | 적당 | 중간 | 중간 | 최상 | 

## 다양한 의미있는 연구와 ResNet 관련 최신 연구들

### Network in Network (NiN)

각 Conv layer 안에 Multi-Layer Perceptron을 쌓는다. FC-layer를 쌓는 것이다.    
맨 처음에 Conv Layer가 있고 그 다음에 FC-layer를 통해 abstract feature를 더 잘 뽑을 수 있게 한다.    
좀 더 복잡한 계층을 만들어서 activation map 을 얻어보자는 아이디어.    
GoogLeNet과 ResNet보다 먼제 Bottleneck 개념을 정립.    
GoogLeNet이 철학적인 영감을 받음.

### Identity Mappings in Deep Residual Networks

ResNet의 저자들은 ResNet의 디자인을 향상시킨 논문을 발표. 여기서 Block design을 조금 향상시킴. direct path를 더 많이 만들었다.

### Wide Residual Network

기존의 ResNet은 깊게 쌓는 것에 열중했지만 중요한 것은 depth가 아니라 변화값이라고 생각. Residual Connection이 있다면 네트워크가 깊어질 필요가 없다.
그래서 Residual block을 넓게 만들었다. Conv layer의 필터 수를 모두 k배 만큼 늘렸다. 그랬더니 기존의 152 layer보다 이런 50 layer가 성능이 더 좋다.
depth대신 filter 수를 늘리면 계산 효율이 증가하는 이점이 있다. 병렬화가 더 잘되기 때문.

### ResNeXt

Residual block 내에 다중 병렬 경로를 추가. 이런 pathways의 총합을 cardinality라고 부름. 하나의 bottleneck ResNet Block은 작지만(4개의 필터) 이런 블럭이 많다(32개). 여러 레이어를 병렬로 묶는 다는 점에서 Inception Module과도 연관이 있다. 

### Deep Networks with Stochastic Depth

ResNet 네트워크가 깊어지면 Vanishing gradient 문제가 발생한다. 그래서 Training time에 몇몇 레이어를 제거한다 (identity connection으로 만든다). 얕은 네트워크는 학습이 잘 되기 때문. Dropout과 유사하다. Test할 때는 모든 레이어를 다 쓴다.

## ResNet 이후에 나온 신경망

### FractalNet

Residual connection이 쓸모 없다고 주장. 얕은 신경망을 깊은 신경망으로 잘 전환하는 것이 핵심이라고 생각. 신경망이 프랙탈 같이 생김. 여기서는 얕은 경로와 깊은 경모를 모두 연결. 학습할 때는 dropout을 써서 일부 경로만 학습. 테스트할 때는 전부 쓴다. 성능이 좋음을 입증. 

### DenseNet

Dense Block이라는 것이 있다. 한 레이어가 그 앞 단의 모든 레이어와 연결되어 있다. 입력이 모든 레이어에 들어가고 모든 레이어의 출력이 합쳐진다. 이렇게 합쳐진 값을 Conv layer에 쓴다. 이 과정에서 깊이를 줄이는 과정이 포함. Dense Connection이 Vanishing gradient 문제를 완화시킬 수 있다 생각.

### 결론

레이어간의 연결을 어떻게 할 지, depth를 어떻게 구성할 지에 관한 연구가 많다.

### SqueezeNet

효율성에 치중한 네트워크. Fire Module이라는 것을 도입. Fire Module의 Squeeze layer는 1x1 필터로 구성되어 있고, 이 출력 값이 1x1, 3x3 필터 들로 구성되어있는 expand layer로 전달됨. AlexNet만큼의 성능을 보이지만 파라미터는 50배 더 적었다. 더 압축하면 500배 더 작아진다. 0.5Mb 밖에 차지하지 않는다.