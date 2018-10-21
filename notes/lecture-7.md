
# Lecture 7 note

hyperparameter를 정할 때 넓은 범위부터 시작한 다음에 성능이 좋은 것을 기준으로 점차 줄여 나가라.
중요한 hyperparameter에는 learning rate가 가장 중요하고 그 다음으로는 regularization, learning rate decay, model size 등이 있다.

## Optimization

### SGD

신경망을 학습하는 가장 중요한 전략은 바로 최적화시키는 것이다.
SGD는 간단하지만 몇 가지 문제가 있다.

1.  만약 loss function이 중요 하지 않은 방향에 민감하고 중요한 방향에 민감하면 중요한 방향으로는 움직이지 않을 수 있다. 지그재그로 움직이면서 중요한 방향으로 간다. 이러한 문제는 차원이 높아질수록 잘 나타난다.
2. local minima나 saddle point에서 SGD는 멈춰버린다. 낮은 차원에서는 local minima가 큰 문제 처럼 보이고 saddle point가 작은 문제 처럼 보이지만 차원이 높아지면 반대가 된다. 높은 차원에서 saddle point가 의미하는 바는 어떤 방향은 loss가 높아지고 어떤 방향은 loss가 낮아진다는 의미이다. 차원이 높으면 이러한 현상은 더 자주 나타나게 된다. local minima가 높은 차원에서 이루어 질려면 모든 방향에서의 local minima가 있어야 하는데 이는 잘 발생하지 않는다. 게다가 saddle point의 근처에서는 progress가 엄청 느리게 진행된다는 문제가 발생한다.
3. loss 값을 구할 때 원래는 모든 training에 대해서 구해야 하지만 이건 너무 비싸므로 mini-batch에 대한 loss만 구한다. 하지만 이는 실제 loss 값과 다를 수 있으므로 최적의 값을 찾아 갈 때 까지 noise가 있게 찾아간다.

Q. 일반 gradient descent를 쓰면 이 모든 문제가 사라지는가?    
A. 1번이나 2번 문제는 그대로 존재한다.

### Momentum

그래서 이런 문제를 해결 하기 위해 Momentum을 추가한다. momentum이라는 작은 분산을 추가한다. 항상 속도를 보관 하고 있다가 gradient가 향하는 방향 보다는 이 속도가 가르키는 방향으로 나아가게 한다.

local minima, saddle point, poor conditioning, gradient noise의 모든 문제를 해결한다.

### Nesterov Momentum

nesterov momentum : 일반적인 momentum은 한 점에서의 velocity와 gradient를 구한 다음에 합하지만 nesterov momentum은 velocity가 향하는 점에서의 gradient를 구한 다음에 합한다.
velocity의 좋은 초기화 값은 0이다. 이는 hyperparameter가 아니다.
이 두 momentum은 속도 때문에 minima를 over estimate를 해서 더 나가버리지만 곧 뒤로 돌아와서 minima로 간다.

Q. 만약 minima가 너무 좁고 깊어서 velocity 때문에 지나쳐버리면 어떻게 하느냐?    
A. 이런 minima는 너무 안 좋은 것이다. 그래서 training set을 늘리면 minima가 완만해 질 것이다.

잘 보면 nesterov가 일반 momentum보다 덜 over estimate 한다.

### AdaGrad

AdaGrad : 학습하는 동안 gradient의 제곱의 합을 저장하고 있는다. velocity 대신 gradient 제곱을 사용한다. gradient가 큰 방향에서는 이 큰 값을 나누어 주므로 좀 천천히 움직이게 하고 작은 방향에서는 그 반대.

step이 진행 되면 진행 될 수록 움직이는게 점차 느려지는데 convex 같은 minima에서는 minima에 가까워 질 수록 점차 느려지는 게 좋으므로 이렇게 하는 것이 좋지만 non-convex minima에서는 saddle point에 갇혀있을 수도 있다.

### RMSProp

RMSProp : grad_squared를 저장하고 있지만 decay_rate를 이용하여 예전 gradient의 영향을 점차 작게 한다.

SGD Momentum은 minima에 대해 overshoot을 하지만 RMSProp는 조금씩 움직인다.
AdaGrad는 쓰지 않는 것을 추천

### Adam

위 두 방법을 합친 것이 Adam
맨 처음에서는 second_moment가 거의 0이 므로 gradient가 엄청 커지는 문제가 생겨버린다. 그러면 아무리 처음에 초기화를 잘해도 이렇게 튀어버려민 의미가 사라져버린다.

Q. 1e-7을 하는 이유?    
A. 0으로 나눠지지 않게 하기 위해서

learing rate가 갈수록 decay하게 하자 momentum에서는 자주 쓰지만 adam 에서는 자주 쓰지 않는다.

## Second-Order Optimization

지금까지 이야기 한 알고리즘은 전부 First-Order Alg이다. 우리는 어떤 한 점의 gradient를 구한 다음에 그 점에 대한 선형 함수 정보를 찾아서 조금 진행하는 것이다. 그런데 그 다음 점의 gradient도 이용하면 근사하는 이차 함수를 만들 수 있는데 이 이차 함수의 최저점으로 바로 이동 할 수 있다. 이는 learning rate가 없다. 초기 버전의 Newton 방법은 learning rate가 없다. 하지만 이 이차 함수 근사 값도 결국은 근사값이기 때문에 learning rate를 이용해서 minima로 한 스템 이동하게 한다. 근데 이건 딥러닝에서 잘 쓰이지 않는다. Hessian matrix가 N^2의 element를 가지고 있고 이를 inverting 하는데 N^3의 컴퓨팅 파워가 든다.

### Quasi-Newton Method (BGFS)

full Hessian과 그 전부를 inverting하는 것을 이용하는 것 보다 근사값을 이용하여 한다.
L-BGFS라고 불리우는 더 낮은 차원의 근사값을 이용하는 것이 더 대중적이다.

### L-BFGS

많은 딥러닝 문제에서 잘 동작하지 않는다.

## Conclusion

그래서 Adam을 많이 이용하고 full batch update할 여유가 있으면 L-BFGS를 해라
L-BFGS는 신경망을 학습할 때는 잘 쓰지 않지만 style-transfer에서는 자주 쓰인다.
style-transfer : less stochasticity, fewer parameter를 가지고 있지만 optimization problem 을 풀고 싶을 때 (?)

## Model Ensembles

지금까지 이야기 한 모든 방법들은 Training Error를 줄이기 위한 것들이다. 이 모든 Optimization Alg는 Training Error를 서서히 낮추고 loss function의 최솟값을 찾는 것이다. 근데 우리는 Training Error를 그리 신경 쓰지 않는다. 우리는 test data의 성능을 중요시 여긴다. 우리는 train과 test 간의 간격을 줄이고 싶다. 우리가 loss function의 최솟값을 잘 찾았으면 이 간격을 줄이는데 무엇을 해야 하는가?

그 중에 한 방법이 Model Ensembles이다. 서로 다른 모델을 이용하여 학습을 한 다음에 test set 에서는 이들의 평균을 이용한다. 2퍼센트 정도 잘 overfitting을 하지 않고 성능이 높아진다. 급격한 성능 향상은 아니지만 성능을 극대화 시키기위해서 일반적으로 쓰이는 방법이다.

여러 개의 모델을 학습하기 보다는 한 모델이 학습되는 와중의 스냅샷을 저장해 놨다가 이를 이용하여 Ensemble을 하는 방법도 있다.

또한 Learing rate를 주기적으로 엄청 높였다가 엄청 낮췄다가 하면 진행 될때 마다 나름 괜찮은 local minima에 안착하는데 이렇게 local minima에 안착한 것들 끼리 ensemble 하면 나름 괜찮다.

Q. Ensemble할 때 hyperparameter를 같게 하는가?    
A. 보통 아니다.

Polyak averaging : 학습 할 때 parameter vector를 decay 하게 하면서 저장해 놨다가 모델이 스스로 Ensemble하게 만든다. 자주 쓰이지는 않는다.

## Regularization

어떻게 한 모델의 성능을 올리는 가? 여러 개의 모델을 학습 한 다음에 Ensemble 하는 것은 비싸므로 별로 좋지 않다. 이에 해당하는 전략이 정규화인데, L2 정규화는 신경망에서는 잘 되지 않는다. 그래서 신경망에서는 다른 걸 쓴다.

### Dropout Regularization

forward pass에서 무작위로 어떤 뉴런의 값을 0으로 만든다. activation 값을 0으로 만든다. 
일반적으로 fully connected layer 에서 쓰지만 convolution layer에서도 쓴다. 무작위로 고른 전체 feature map을 0으로 만든다. 한 채널을 0으로 만든다.
dropout이 동작하는 이유 1 : 마지막에 decision을 하는 레이어에서 똑같은 가중치를 가져야 하는 feature들 중에 하나에 가중치가 쏠리는 현상을 막는다.
dropout이 동작하는 이유 2 : 한 모델에서 ensemble을 하는 꼴이다.

test 할 때 하면 어떻게 되느냐? 같은 사진이 서로 다르게 분류 되는 것은 실제로 모델을 쓸때 이상하므로 test할 때는 dropout을 하지 않는다.
확률적으로 따지면 activation value가 1/2가 되므로 dropout을 하지 않고 그냥 각 activation value를 절반으로 만든다.

Train 할 때는 forward pass에서 단순히 2줄을 추가 함으로써 dropout을 할 수 있고
Test 할 때는 확률을 곱함으로써 할 수 있다.

Inverted dropout : test할 때 성능을 중요시 여기면 확률을 곱하는 것을 없애도 된다. 자주 쓰인다.
training 할 때는 더 오래 걸린다(?)

신경망에서의 정규화를 일반화 시키면 training 할 때는 무작위성을 추가하고 test할 때는 이 무작위성을 평균을 내서 하면 된다.
그런데 batch normalization이 이런 작용을 한다. 그래서 dropout을 하지 않아도 된다.

### Data Augmentation

이미지를 무작위로 조금씩 변형해서 라벨을 그대로 유지 되도록 하

1\. Random Horizontoal Flip

2\. Random Crops and Scales

Training할 때 무작위 scale을 가지고 무작위 crop을 한다.
Test할 때는 무작위 scale과 crop값을 평균을 낸 것을 이용한다. 일반적으로 중앙과 4방향의 코너 그리고 각각의 flip을 이용한다.

3\. Colo Jitter

PCA 방향을 적용해서 RGB를 jitter 한다.

### DropConnect

activation value를 0으로 만드는 것이 아니라 weight을 일부를 무작위로 0으로 만든다.

### Fractional Max Pooling

pooling layer에서 pooling할 부분을 랜덤으로 지정해서 한다.

### Stochastic Depth

학습할 때는 무작위로 어떤 layer를 건너 뛴다. 테스트 할 때는 모든 layer를 쓴다.

Q. 여러 개의 Regularization을 쓰는가?    
A. 일반적으로 Batch Normalization을 쓰면 대부분 잘 되는데 만약에 이걸로 충분하지 않을 것 같으면 더 넣는다.

## Trasfer Learning

다른 사람이 training 해놓은 모델을 그대로 가져와서 classification을 하는 제일 마지막 layer를 빼고 모두 freeze 하고 이 모델을 학습한다.
만약 데이터가 더 있다면 freeze를 하지 않는 layer 수를 늘리면 된다.

기존과 비슷한 데이터 셋    
적은 데이터를 가지고 있는 경우 : 가장 마지막 layer에 linear classifier를 적용한다.    
많은 데이터를 가지고 있는 경우 : 여러 마지막 layer를 학습시킨다.    

기존과 다른 데이터 셋    
적은 데이터를 가지고 있는 경우 : 여러 다양한 시도를 해라
많은 데이터를 가지고 있는 경우 : 더 많은 layer를 학습시킨다.

이거는 다른 구조에 잘 스며들 수 있으므로 이미지 처리를 하는 CNN 레이어의 경우 자기가 직접 학습하는 것이 아니라 ImageNet의 잘 되어 있는 것을 가져다가 쓴다.
무슨 문제를 해결하려고 하던지 데이터 셋이 많이 없으면 그냥 학습되어 있는 모델을 다운로드 받아서 해라