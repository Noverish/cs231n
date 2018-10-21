
# Lecture 8 note

## CPU vs GPU

CPU : Central Processing Unit
GPU : Graphics Processing Unit

딥러닝에서는 NVIDIA가 AMD보다 더 널리 쓰인다. AMD가지고 있는 사람은 딥러닝할 때 조금 힘들거다.

GPU는 행렬 곱에 강하다.

cuDNN이 unoptimized CUDA보다 빠르다.

## Deep Learing Frameworks

사용하는 이유 3가지
1. 엄청 복잡한 계산 그래프를 직접 만들지 않아도 된다.
1. forward pass만 잘 만들어 놓으면 back propagation 은 알아서 만들어진다.
1. GPU를 효율적으로 사용할 수 있다.

Numpy는 CPU에서만 동작한다.