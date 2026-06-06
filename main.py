import sys
sys.stdout.reconfigure(encoding='utf-8')   # 한국어 출력 깨짐 방지

import numpy as np
import tensorflow as tf

# 물리 시뮬레이션: 한 스텝 진행
def simulate_step(state, dt=0.1, g=9.8):
    """공의 현재 상태에서 dt 시간 뒤 상태 계산."""
    x, y, vx, vy = state
    
    x_new  = x + vx * dt
    y_new  = y + vy * dt
    vx_new = vx
    vy_new = vy - g * dt
    
    return (x_new, y_new, vx_new, vy_new)

# 모델 학습을 위한 데이터 생성
def generate_data(num_samples=10000):
    """학습데이터 만들기. 각 샘플: 랜덤상태 → 한 스텝 뒤 상태"""
    X = []
    y = []

    for _ in range(num_samples):
        # 랜덤 초기 상태 (실제 사용 범위 다 커버하게 넓힘)
        state = (
            np.random.uniform(-30, 30),    # x:  넓게
            np.random.uniform(-50, 50),    # y:  음수도 (떨어진 후)
            np.random.uniform(-15, 15),    # vx: 넓게
            np.random.uniform(-40, 40),    # vy: 크게 ← 핵심 (떨어지는 vy 포함)
        )
        
        # 한 스텝 진행
        next_state = simulate_step(state)
        
        X.append(state)
        y.append(next_state)
    
    return np.array(X), np.array(y)

# 데이터 생성
print("데이터 만드는 중...")
X, y = generate_data(30000)   # 데이터 3배 (넓은 범위 다 커버하려면 더 많은 샘플)
print(f"X 모양: {X.shape}")   # 예상: (10000, 4)
print(f"y 모양: {y.shape}")   # 예상: (10000, 4)
print(f"첫 샘플 - 입력: {X[0]}")
print(f"첫 샘플 - 출력: {y[0]}")

# 신경망 만들기 (강화: 32→64, 2층→3층)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden 1: 64
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden 2: 64
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden 3: 64 (추가)
    tf.keras.layers.Dense(4),                       # 출력: 4개
])

# 모델 요약 출력
model.summary()

# 컴파일
model.compile(
    optimizer='adam',
    loss='mse'
)

# 학습
print("\n학습 시작...")
history = model.fit(
    X, y,
    epochs=100,            # 50 → 100 (더 오래 학습)
    batch_size=64,         # 32 → 64
    validation_split=0.1,
    verbose=1
)

print("\n학습 완료!")
print(f"최종 train loss: {history.history['loss'][-1]:.6f}")
print(f"최종 val loss:   {history.history['val_loss'][-1]:.6f}")

# === 검증 1: 단일 예측 비교 ===
print("\n=== 검증 1: 단일 예측 vs 실제 물리 ===")

# 테스트할 임의 상태
test_state = np.array([[5.0, 10.0, 2.0, 3.0]])   # 2D 배열 (모델이 batch 형태 받음)

# 실제 물리로 한 스텝
real_next = simulate_step(test_state[0])

# 신경망 예측
pred_next = model.predict(test_state, verbose=0)[0]

print(f"입력 상태:    {test_state[0]}")
print(f"실제 물리:    {np.array(real_next)}")
print(f"신경망 예측: {pred_next}")
print(f"오차 (절대): {np.abs(np.array(real_next) - pred_next)}")

# === 검증 2: 궤적 시뮬레이션 ===
import matplotlib.pyplot as plt

print("\n=== 검증 2: 궤적 비교 (50 스텝) ===")

# 시작 상태: (0,0)에서 위로 던지기
initial_state = np.array([0.0, 0.0, 3.0, 10.0])

# 실제 물리로 시뮬레이션
real_traj = [initial_state.copy()]
state = initial_state.copy()
for _ in range(50):
    state = np.array(simulate_step(tuple(state)))
    real_traj.append(state.copy())
real_traj = np.array(real_traj)

# 신경망으로 시뮬레이션 (자기 예측을 또 입력으로)
pred_traj = [initial_state.copy()]
state = initial_state.copy()
for _ in range(50):
    state = model.predict(state.reshape(1, 4), verbose=0)[0]
    pred_traj.append(state.copy())
pred_traj = np.array(pred_traj)

# 그림
plt.figure(figsize=(10, 6))
plt.plot(real_traj[:, 0], real_traj[:, 1], 'b-', label='Real Physics', linewidth=2, alpha=0.7)
plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', label='Neural Network', linewidth=2)
plt.scatter(0, 0, color='green', s=150, marker='*', label='Start', zorder=5)
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Ball Trajectory: Physics vs Neural Network')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("그림 창 닫으면 종료됩니다.")
