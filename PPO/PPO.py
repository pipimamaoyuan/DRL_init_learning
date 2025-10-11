import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
import time
from matplotlib import rcParams


rcParams['font.sans-serif'] = ['SimSun'] # 使用宋体
rcParams['axes.unicode_minus'] = False # 解决负号 '-' 显示为方块的问题

# 设置随机种子以保证结果可复现
torch.manual_seed(0)
np.random.seed(0)
# 如果可用则启用CUDA随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# 设备选择（优先使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_Training_Episodes = 5000  # 最大训练回合数
MAX_Training_Steps = 500      # 每回合最大步数

MAX_Test_Episodes = 10        # 最大测试回合数

MAX_demo_Episodes = 3         # 最大演示回合数

class ActorCritic(nn.Module):
    """
    Actor-Critic 网络
    - Actor: 输入状态，输出动作的概率分布
    - Critic: 输入状态，输出状态的价值估计
    CartPole-v1是离散动作空间（左/右），所以使用Categorical分布
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # 共享的特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Actor网络 - 输出动作的概率分布
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic网络 - 输出状态的价值
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self):
        """前向传播不直接使用，通过具体方法获取动作和价值"""
        raise NotImplementedError
        
    def get_action(self, state):
        """
        根据状态选择动作
        返回: 动作，动作的对数概率，动作的概率分布的熵
        """
        # 通过共享层提取特征
        hidden = self.shared_layers(state)
        
        # Actor输出动作的logits（未归一化的概率）
        logits = self.actor(hidden)
        
        # 创建分类分布（离散动作）
        action_dist = Categorical(logits=logits)
        
        # 从分布中采样一个动作
        action = action_dist.sample()
        
        # 计算该动作的对数概率（用于PPO损失计算）
        log_prob = action_dist.log_prob(action)
        
        # 计算分布的熵（用于鼓励探索）
        entropy = action_dist.entropy()
        
        return action.item(), log_prob, entropy # 返回动作，动作的对数概率，动作的概率分布的熵
        
    def get_value(self, state):
        """估计状态的价值"""
        hidden = self.shared_layers(state)
        value = self.critic(hidden)
        return value
        
    def evaluate_actions(self, state, action):
        """
        评估给定状态下采取特定动作的概率和价值
        用于PPO更新阶段
        """
        hidden = self.shared_layers(state)
        
        # 计算动作的概率
        logits = self.actor(hidden) 
        action_dist = Categorical(logits=logits) # 创建分类分布
        log_prob = action_dist.log_prob(action) # 计算给定动作的对数概率
        entropy = action_dist.entropy() # 计算分布的熵
        
        # 计算状态价值
        value = self.critic(hidden)
        
        return log_prob, entropy, value # 返回动作的对数概率，熵，状态价值

class PPO:
    """
    PPO算法实现类
    包含PPO算法的核心组件：经验收集、GAE计算、损失函数、模型更新等
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 clip_epsilon=0.2, ppo_epochs=4, batch_size=64):
        # 初始化参数
        self.gamma = gamma  # 折扣因子
        self.clip_epsilon = clip_epsilon  # PPO裁剪参数
        self.ppo_epochs = ppo_epochs  # PPO更新轮数
        self.batch_size = batch_size  # 小批量大小
        # 初始化Actor-Critic网络（并移动到设备）
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.device = device

        # 优化器
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # 存储训练数据的缓冲区
        self.states = []       # 状态
        self.actions = []      # 动作
        self.log_probs = []    # 动作的对数概率
        self.rewards = []      # 奖励
        self.values = []       # 状态的价值
        self.dones = []        # 是否结束

    def get_action(self, state):
        """
        Wrapper that delegates action selection to the underlying ActorCritic.
        Accepts a torch tensor `state` and returns (action, log_prob, entropy),
        matching the ActorCritic.get_action contract.
        """
        # 将 state 转换为 tensor 并移动到正确设备
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)
        return self.actor_critic.get_action(state)

    def get_value(self, state):
        """
        Wrapper that delegates value estimation to the underlying ActorCritic.
        Returns a torch tensor containing the state value.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)
        return self.actor_critic.get_value(state)
        
    def store_transition(self, state, action, log_prob, value, reward, done):
        """存储单步转移数据"""
        self.states.append(state)  # 状态
        self.actions.append(action)  # 动作
        self.log_probs.append(log_prob)  # 动作的对数概率
        self.values.append(value)  # 状态的价值
        self.rewards.append(reward)  # 奖励
        self.dones.append(done)  # 是否结束
        
    def clear_buffer(self):
        """清空经验缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def compute_advantages(self, next_value):
        """
        计算广义优势估计(GAE)
        GAE结合了TD误差的多步估计，平衡偏差和方差
        """
        # 将数据转换为tensor并移动到设备
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)

        # 添加下一个状态的价值（保持为tensor，避免不必要的CPU转换）
        if isinstance(next_value, torch.Tensor):
            next_val = next_value.squeeze().detach()
            if next_val.device != self.device:
                next_val = next_val.to(self.device)
        else:
            next_val = torch.tensor([float(next_value)], dtype=torch.float32, device=self.device).squeeze()

        values = torch.cat([values, next_val.unsqueeze(0)])
        
        # 计算TD误差 δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        
        # 计算GAE优势：A_t = Σ (γλ)^l * δ_{t+l}
        # 这里我们使用λ=0.95，这是PPO中常用的值
        advantages = []
        advantage = 0.0
        lambda_param = 0.95
        
        # 反向计算优势（从最后一步开始）
        for delta in reversed(deltas):
            advantage = delta + self.gamma * lambda_param * advantage
            advantages.insert(0, advantage)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # 计算回报（用于价值函数目标）并移动到设备
        returns = advantages + torch.tensor(self.values, dtype=torch.float32, device=self.device)

        return advantages, returns
    
    def update(self, next_state):
        """
        PPO核心更新步骤
        包括：计算优势、多轮小批量更新、裁剪目标函数
        """
        # 计算下一个状态的价值（用于TD误差计算）
        with torch.no_grad():
            # 将下一个状态移动到设备并计算价值
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
            next_value = self.actor_critic.get_value(next_state) # 估计下一个状态的价值
        
        # 计算优势估计和回报
        advantages, returns = self.compute_advantages(next_value)

        # 标准化优势（减少方差，提高训练稳定性）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 将存储的数据转换为tensor并移动到设备
        states = torch.tensor(self.states, dtype=torch.float32, device=self.device) # 状态
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device) # 动作
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device) # 旧动作的对数概率

        # 多轮PPO更新
        for epoch in range(self.ppo_epochs):
            # 随机打乱数据索引
            indices = np.arange(len(self.states))
            # 在 PPO 的更新阶段，通常会进行多轮（ppo_epochs）训练，
            # 每轮都希望以不同顺序使用经验数据，避免模型对数据顺序过拟合（比如总是先看到“成功”的轨迹）。
            np.random.shuffle(indices)
            
            total_actor_loss = 0.0
            total_critic_loss = 0.0
            total_entropy = 0.0
            num_batches = 0
            
            # 小批量更新
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # 获取小批量数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算当前策略的动作概率和价值
                batch_new_log_probs, entropy, values = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions) # 评估动作，获取新的对数概率，熵，状态价值
                
                # 计算概率比 r(θ) = π_new(a|s) / π_old(a|s) = exp(log_prob_new - log_prob_old)
                ratios = torch.exp(batch_new_log_probs - batch_old_log_probs)
                
                # PPO-Clip 目标函数的核心部分
                # 第一部分：概率比 乘以 优势
                surr1 = ratios * batch_advantages
                # 第二部分：裁剪后的 概率比 乘以 优势
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                # PPO目标：取两者中的较小值，防止更新幅度过大
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic损失：价值函数 与 回报 的均方误差
                critic_loss = (values.squeeze() - batch_returns).pow(2).mean()
                
                # 熵奖励：鼓励探索，防止策略过早收敛
                entropy_bonus = -entropy.mean()
                
                # 总损失 = Actor损失 + Critic损失 + 熵奖励
                # 系数0.5和0.01是常用的超参数
                total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_bonus
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                
                self.optimizer.step()
                
                # 累加损失（用于平均）
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1
            
            # # 打印当前 epoch 的平均损失
            # if num_batches > 0:
            #     avg_actor = total_actor_loss / num_batches
            #     avg_critic = total_critic_loss / num_batches
            #     avg_entropy = total_entropy / num_batches
            #     print(f"  PPO Epoch {epoch+1}/{self.ppo_epochs}: "
            #         f"actor_loss={avg_actor:.4f}, "
            #         f"critic_loss={avg_critic:.4f}, "
            #         f"entropy={avg_entropy:.4f}")
            
        # 清空经验缓冲区
        self.clear_buffer()
        
    def save_model(self, file_path):
        """保存模型参数"""
        # 在保存时将参数移动到CPU，保证跨设备可加载
        cpu_state = {k: v.cpu() for k, v in self.actor_critic.state_dict().items()}
        torch.save(cpu_state, file_path)
        print(f"模型已保存到: {file_path}")
        
    def load_model(self, file_path):
        """加载模型参数"""
        # 使用 map_location 以支持在CPU/GPU之间加载
        state_dict = torch.load(file_path, map_location=self.device)
        self.actor_critic.load_state_dict(state_dict)
        # 确保模型被移动到当前设备
        self.actor_critic.to(self.device)
        print(f"模型已从 {file_path} 加载 到 {self.device}")

def train_ppo(env_name="CartPole-v1", num_episodes=MAX_Training_Episodes, max_steps=MAX_Training_Steps, 
               model_save_path="PPO/ppo_cartpole"):
    """
    训练PPO算法的主函数
    """
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"环境: {env_name}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 初始化PPO算法
    ppo_agent = PPO(state_dim, action_dim)
    
    # 记录训练过程中的回报
    episode_rewards = []
    moving_average_rewards = []
    best_reward = -float('inf')
    
    # 训练循环
    for episode in range(num_episodes):
        state, _ = env.reset() # 重置环境，获取初始状态
        episode_reward = 0
        
        # 清空经验缓冲区（为新的回合做准备）
        ppo_agent.clear_buffer()
        
        # 与环境交互，收集经验
        for step in range(max_steps):
            # 将状态转换为tensor
            state_tensor = torch.FloatTensor(state)
            
            # 使用当前策略选择动作
            with torch.no_grad():
                action, log_prob, _ = ppo_agent.get_action(state_tensor) # 获得动作及其对数概率
                value = ppo_agent.get_value(state_tensor) # 估计当前状态的价值
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储转移数据
            ppo_agent.store_transition(state, action, log_prob, value.item(), reward, done)

            # 更新状态和累计奖励
            state = next_state
            episode_reward += reward
            
            # 如果回合结束或达到最大步数，进行PPO更新
            if done or step == max_steps - 1:

                # PPO更新
                ppo_agent.update(next_state)
                break
        
        # 记录回报
        episode_rewards.append(episode_reward)
        
        # 计算移动平均回报（平滑曲线）
        if len(episode_rewards) >= 10:
            moving_avg = np.mean(episode_rewards[-10:])
        else:
            moving_avg = np.mean(episode_rewards)
        moving_average_rewards.append(moving_avg)
        
        # 更新最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            ppo_agent.save_model(model_save_path + "_best.pth")
        
        # 打印训练进度
        if episode % 10 == 0:
            print(f"回合 {episode}, 回报: {episode_reward}, 移动平均回报: {moving_avg:.2f}")

        # 提前停止条件：连续100个回合平均回报达到200
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 200:
            print(f"训练完成！在回合 {episode} 达到目标回报")
            break
    
    # 保存最终模型
    ppo_agent.save_model(model_save_path + "_best.pth")
    
    # 关闭环境
    env.close()
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.6, label='Rewards')
    plt.plot(moving_average_rewards, linewidth=2, label='Rewards (Moving Average)')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('PPO Training Curve on CartPole-v1')
    plt.legend()
    plt.grid(True)
    plt.savefig('PPO/ppo_training_curve.png')
    plt.show()
    
    return ppo_agent, episode_rewards

def test_ppo(env_name="CartPole-v1", model_path="PPO/ppo_cartpole_best.pth", num_episodes=MAX_Test_Episodes, render=True):
    """
    测试训练好的PPO模型
    """
    # 创建环境
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)

    state_dim = env.observation_space.shape[0] # 状态维度
    action_dim = env.action_space.n # 动作维度
    
    # 初始化PPO算法并加载模型
    ppo_agent = PPO(state_dim, action_dim)
    ppo_agent.load_model(model_path)
    
    test_rewards = []
    
    print("开始测试...")
    for episode in range(num_episodes):
        state, _ = env.reset() # 重置环境，获取初始状态
        episode_reward = 0
        steps = 0
        
        while True:
            # 将状态转换为tensor
            state_tensor = torch.FloatTensor(state)
            
            # 使用训练好的策略选择动作（不需要梯度）
            with torch.no_grad():
                action, _, _ = ppo_agent.get_action(state_tensor)
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            real_done = done or truncated
            
            # 更新状态和累计奖励
            state = next_state
            episode_reward += reward
            steps += 1
            
            # 如果回合结束
            if real_done:
                break
        
        test_rewards.append(episode_reward)
        print(f"测试回合 {episode+1}: 回报 = {episode_reward}, 步数 = {steps}")
    
    # 计算平均测试回报
    avg_reward = np.mean(test_rewards)
    print(f"\n平均测试回报: {avg_reward:.2f}")
    
    env.close()
    
    return test_rewards

def demo_agent(env_name="CartPole-v1", model_path="PPO/ppo_cartpole_best.pth", num_demos=MAX_demo_Episodes):
    """
    展示智能体在环境中的表现
    """
    env = gym.make(env_name, render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 初始化PPO算法并加载模型
    ppo_agent = PPO(state_dim, action_dim)
    ppo_agent.load_model(model_path)
    
    print("开始演示...")
    for demo in range(num_demos):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # 将状态转换为tensor
            state_tensor = torch.FloatTensor(state)
            
            # 使用训练好的策略选择动作
            with torch.no_grad():
                action, _, _ = ppo_agent.get_action(state_tensor)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 更新状态和累计奖励
            state = next_state
            total_reward += reward
            steps += 1
            
            # 添加小延迟以便观察
            time.sleep(0.01)
            
            # 如果回合结束
            if done:
                print(f"演示 {demo+1}: 总回报 = {total_reward}, 步数 = {steps}")
                break
    
    env.close()

if __name__ == "__main__":
    # 训练PPO算法
    print("开始训练PPO算法...")
    ppo_agent, training_rewards = train_ppo(num_episodes=MAX_Training_Episodes)
    
    # 测试训练好的模型
    print("\n测试训练好的模型...")
    test_rewards = test_ppo(num_episodes=MAX_Test_Episodes, render=False)
    
    # 展示智能体表现
    print("\n展示智能体表现...")
    demo_agent(num_demos=MAX_demo_Episodes)
    
    # 打印训练统计信息
    print(f"\n训练统计:")
    print(f"总训练回合数: {len(training_rewards)}")
    print(f"最高回合回报: {max(training_rewards)}")
    print(f"平均最后10回合回报: {np.mean(training_rewards[-10:]):.2f}")
    print(f"平均最后100回合回报: {np.mean(training_rewards[-100:]):.2f}")
    print(f"平均测试回报: {np.mean(test_rewards):.2f}")