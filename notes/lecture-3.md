
# Lecture 3 note

## loss function

## multi-class SVM loss

### loss function (hinge loss)
![image001](https://raw.githubusercontent.com/Noverish/cs231n/master/notes/images/001.png)

Q. S가 뭐고 S_Y_i가 뭐냐    
A. Y_i는 실제 정답의 카테고리 이다. S_Y_i는 i번째 이미지의 정답 클래스의 스코어이다.

Q. safty margin이 1인걸 어떻게 정하냐    
A. 우리는 스코어이 실제 값을 중요하게 여기는 것이 아니라 값들의 상대적인 차이에 신경을 쓴다. 하다보면 1은 

Q1 : 차 이미지의 점수들을 조금 조정해도 loss 점수는 변하지 않는다.

Q2 : SVM loss의 최솟값은 0, 최댓값은 무한.

Q3 : 만약 W를 초기화 할 때 모든 클래스의 값들이 거의 0에 가깝게 하면 loss 값은 클래스 수 - 1

Q4 : SVM loss는 정답인 클래스는 빼고 더 했는데 포함해서 더하면 loss가 1 증가한다.    
1이 더 해진다고 다른 분류기가 학습 되는 것이 아니다. 그냥 관례에 따라 loss의 최솟값을 0으로 맞춰주는 것.

Q5 : loss 구할 때 합을 구하는 것 말고 평균을 구한다고 해도 별로 달라지지 않는다. 그저 loss 값의 스케일만 달라질 뿐.

Q6 : loss 구할 때 제곱 항으로 만들어 버리면 loss 값이 크게 바뀐다.

Q. 왜 제곱 항을 고려하느냐    
A. loss에 제곱을 하면 그냥 안 좋은게 곱절로 안 좋아 진다고 가정하는 것이다.

loss가 0이 되게 하는 W는 유일하지 않다. 2W도 loss를 0으로 만든다.

## Regularization

손실 함수에 model이 얼마냐 단순 하냐에 대한 값도 넣는다.

L2 regularization : W에 대한 Euclidean Norm(Squared Norm). 미분할 때 깔끔하라고 앞에 1/2를 붙이기도 한다.
L1 regularization : W가 희소행렬이 되도록 한다.

X = [1, 1, 1, 1]    
W1 = [1, 0, 0, 0]    
W2 = [0.25, 0.25, 0.25, 0.25]    
가 있으면 L2는 W2를 선호하고 L1은 W1을 선호한다.

L2    
W가 고르게 분포되어 있으면 간단하다고 생각.
모든 X의 값들이 비슷하게 결과에 영향을 미쳤으면 좋겠다고 생각.

L1    
W에 0이 많으면 간단하다고 생각.
일반적으로 sparse한 W를 선호한다. W에 대부분이 0이 되게 한다.

## Softmax (Multinormial logistic regression)

SVM loss는 스코어 자체에 대한 해석은 하지 않았다.
하지만 이거는 스코에 자체에 추가적인 의미를 부여한다. 이 스코어를 가지고 클래스 별 확률 분포를 계산한다.

### softmax function
![image002](https://raw.githubusercontent.com/Noverish/cs231n/master/notes/images/002.png)

### loss function (corss-entropy loss)
![image003](https://raw.githubusercontent.com/Noverish/cs231n/master/notes/images/003.png)
![image004](https://raw.githubusercontent.com/Noverish/cs231n/master/notes/images/004.png)

Q1. Softmax loss의 최댓값은 무한, 최솟값은 0. 근데 컴퓨터로는 절대 이 값에 도달하지 못 한다.

Q2. 스코어들이 0에 근접한 값일 때 loss는 -log(1/C)

## 둘의 차이

SVM은 일정 선만 넘기면 더이상 성능개선에 신경을 쓰지 않는다.
하지만 softmax는 계속해서 노력한다.

## Optimization

## SGD

training set이 너무 크면 loss를 계산하는데 너무 오래걸린다.
그래서 minibatch라는 작은 학습 샘플 집합으로 나눠서 학습.

## 데이터 변환 (?)

### Color Histogram

### HoG

### Bag of Words