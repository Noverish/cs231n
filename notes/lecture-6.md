
# Lecture 6 notes

## Activation function

### 1. Sigmoid

들어오는 input을 0에서 1사이의 값으로 짜낸다.
이 함수의 모양을 뉴런의 활성화 모델(?)로 해석할 수 있다.

Sigmoid에는 3가지 문제점이 있다.

1\. Saturated neuron은 gradient를 없애버린다.

x가 10이거나 -10이면 gradient가 거의 0이 되버린다.
그러면 back propagation에서 이 앞에 있는 레이어에게 gradient decent를 할 수 없다.

2\. Output이 zero center가 아니다.

만약 X가 모두 양수라면 W의 local gradient가 모두 양수거나 모두 음수가 되버린다.
그러면 W의 값을 모두 증가 시키거나 모두 감소시키기 때문에 적잘한 값을 찾을 수 없게 되버린다.

3\. 지수 함수가 계산하기 비싸다.

### 2. tanh

시그모이드와 비슷하게 생겼다. 들어오는 input을 -1과 1사이의 값으로 짜낸다.
zero-centered 이다. 하지만 sigmoid의 1번째 문제는 그대로 가지고 있다.

### 3. ReLU

양수 부분에서는 saturate하지 않다. 이것은 큰 강점이다.
계산하기도 쉽다. 그래서 엄청 빠르다. 6배 정도 빠르다.
AlexNet(2012)에서 쓰이기 시작했다.
zero-centered가 아닌 문제가 있다.
음수 부분에서는 saturate 한다.
그래서 dead ReLU문제가 있다. 만약 처음 시작할 때 들어온 값이 음수면 이 ReLU는 절대 update 하지 않는다.
그래서 사람들은 ReLU neuron을 initialize할 때 작은 양수 값을 넣어준다.

### 4. Leaky ReLU

전혀 saturate하지 않다. ReLU와 마찬가지로 계산하기 쉽다.

### 5. PReLU

Leaky ReLU의 음수 부분의 기울기를 parameter화 해서 이것도 back propagation에 포함시켜버린다.

### 6. ELU

ReLU의 모든 강점을 가지고 있다.
거의 zero-centered 이다. (이건 Leaky ReLU의 강점)
Leaky ReLU 와 달리 음수 부분에서는 saturate 한다. 그래서 noise에 강하다.

### 7. Maxout Neuron

ReLU와 Leaky ReLU의 일반화 버전
saturate하지 않고 죽지 않는다.
W1과 W2를 둘 다 학습 해야 한다.

### Conclusion

ReLU를 쓰고, Leaky ReLU/Maxout/ELU는 한 번 해보고, tanh는 기대하지 말고, sigmoid는 죽어도 쓰지 말아라

## Data Preprocessing

 - zero mean, normalize가 있다.
 - zero mean을 왜 해야 하느냐? 위에서 했듯이 x의 값이 모두 양수면 W의 local gradient가 모두 음수가 되거나 모두 양수가 되버린다.
 - normalize해야 모든 feature들이 똑같은 기여도를 가지고 시작할 수 있다.
 - 이미지는 zero centering은 하는데 normalize는 하지 않는다. 각각의 픽셀값이 나름 비슷한 크기와 분포를 가지고 있기 때문에 일반적인 기계학습에 쓰이는 데이터와 달리 정규화의 필요가 없다.
 - 이미지를 zero mean하려는 경우 모든 사진의 zero mean 이미지를 구하고 이를 각각의 이미지에서 빼준다. 각각의 색깔 채널마다 이걸 해줄 수도 있다.
 - zero mean은 test할 때도 한다.
 - zero mean이 Sigmoid의 문제를 해결해 주는가? -> 첫 레이어에서는 해결해 주지만 그 다음은 그렇지 않다.

## Weight Initialization

모든 W를 0으로 만들어 버리면 모든 W의 값들이 같아져 버린다. 그래서 작은 랜덤 값으로 init한다.    
작은 네트워크에서는 괜찮지만 네트워크가 깊어지면 문제가 생긴다.    
=> 깊어질 수록 히든 레이어의 결과값의 표준 편자가 급격이 0이 되버린다. 마찬가지로 gradient도 모두 0이 되버린다.

그럼 처음 시작할 때 W의 값을 크게 잡아주면 되지 않을까?    
=> 그럼 saturate 해진다. 모든 값이 -1 아니면 1이 되버린다.

Xavier Initialization을 하자 (실험적으로 알아낸 것)    
근데 이건 ReLU에서는 또 먹히지가 않는다. 그래서 2를 나누자

## Batch Normalization

???

## Babysitting the Learning Process

어떻게 training을 모니터링 하고 어떻게 hyperparameter를 결정하는가
1. 제일 먼저 해야 하는 것은 우리 데이터를 preprocessing 하는 것이다.
2. 그 다음은 아키텍쳐를 선택하는 것이다. 히든 레이어가 몇 개고 각 레이어에 몇 개의 뉴런이 있는지 등
3. 우리 네트워크를 초기화 한 다음에 regularization을 끄고 학습 안하고 바로 나온 loss가 납득할 수 있는 값인지를 체크
4. regularization킨 다음에 loss를 체크해서 loss가 커지지 않나 체크 (커지면 정상) (?)
5. 수 십개의 적은 training set을 가지고 regularization을 끈 다음에 학습해서 나온 loss가 0이 되는지 체크 (overfitting이 되야 정상)
6. 이제 regularization를 키고 어떤 learning rate가 적당한 지를 작은 값 부터 시작해서 찾는다. loss값이 거의 변하지 않으면 learning rate를 키운다. 근데 여기서 loss값이 거의 변하지 않는데 training, validation accuracy가 20퍼로 빠르게 점프해버렸다. 어떤 경우에 이런 일이 발생하는 가? (softmax라고 가정) => 확률들이 여전히 분산 되어 있어서 loss값은 여전히 비슷하지만 이 확률들이 조금씩 옳은 방향으로 움직여서 정확도가 서서히 오르는 것이다.
7. 만약 cost가 NaN이면 learning rate가 너무 큰 것이다. cost가 너무 커져서 터져버린 거다. 보통 lr은 1e-3 ~ 1e-5 이다.

## Hyperparameter Optimization

 - cross-validation 젼략을 써서 한다.
 - 조금만 epoch를 돌려 봐서 이 param이 잘 굴러가나 확인한다. cost가 원래 보다 3배 넘어 버리면 이건 아닌거다.
 - 만약 성능이 잘 나온 값의 hyperparameter 범위의 경계 쪽에 있다면 경계를 넓혀서 다시 해주자.
 - random search vs grid search => random이 낫다. 성능에 영향을 끼치는 hyperparameter의 값이 더 다양해지기 때문에
 - loss가 거의 변하지 않다가 어느 순간 급격히 낮아지면 초기화를 잘못한 것이다. => 다시 시작하라
 - Training accuracy가 Validation accuracy보다 매우 높으면 overfitting이 된 것이므로 regularization value를 높이다
 - Training accuracy가 Validation accuracy와 거의 같으면 model capacity를 높인다(?)
 - dW의 값을 트래킹 해야 한다 => dW / W가 0.001정도 나와야 좋다 (이건 코드를 봐야 이해가 빠르다)
