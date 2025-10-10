import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt
import os

# 设置随机种子，保证结果可重现
torch.manual_seed(42)
np.random.seed(42)

MAX_Train_EPISODES = 500
MAX_Train_STEPS = 500

Test_EPISODES = 5
Test_STEPS = 500

class Actor(nn.Module):
    """
    Actor网络：输入状态，输出确定性动作
    在DDPG中，Actor网络负责学习策略，即给定状态选择最优动作
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        # 网络层定义
        self.layer1 = nn.Linear(state_dim, 400)  # 第一层全连接层
        self.layer2 = nn.Linear(400, 300)        # 第二层全连接层
        self.layer3 = nn.Linear(300, action_dim) # 输出层，输出动作值
        
        self.max_action = max_action  # 动作的最大值，用于将输出缩放到合适范围
    
    def forward(self, state):
        """
        前向传播过程
        输入：状态state
        输出：确定性动作，范围在[-max_action, max_action]之间
        """
        # 使用ReLU激活函数，增加非线性表达能力
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        
        # 使用tanh激活函数将输出限制在[-1, 1]范围内，然后乘以max_action进行缩放
        action = self.max_action * torch.tanh(self.layer3(x))
        return action


class Critic(nn.Module):
    """
    Critic网络：
    输入：状态 和 动作；
    输出：Q值（状态-动作价值）
    在DDPG中，Critic网络负责评估Actor网络选择的动作的好坏
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # 第一层：处理状态信息
        self.layer1 = nn.Linear(state_dim, 400)
        
        # 第二层：将状态信息和动作信息合并处理
        self.layer2 = nn.Linear(400 + action_dim, 300)
        
        # 输出层：输出Q值（标量）
        self.layer3 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        """
        前向传播过程
        输入：状态state和动作action
        输出：Q值，表示在给定状态下执行给定动作的价值
        """
        # 处理状态信息
        x = F.relu(self.layer1(state))
        
        # 将状态特征和动作信息拼接起来
        # 注意：这里需要确保state和action在批量维度上一致
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.layer2(x))
        
        # 输出Q值，不需要激活函数，因为Q值可以是任意实数
        q_value = self.layer3(x)
        return q_value


class ReplayBuffer:
    """
    经验回放缓冲区
    用于存储智能体与环境交互的经验(s, a, r, s', done)
    通过随机采样打破数据间的相关性，提高学习稳定性
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 使用双端队列实现固定大小的缓冲区
    
    def push(self, state, action, reward, next_state, done):
        """
        向缓冲区中添加一条经验
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        从缓冲区中随机采样一个批量的经验
        """
        # 随机选择batch_size个经验
        batch = random.sample(self.buffer, batch_size)
        
        # 将经验数据分别提取并转换为numpy数组
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return state, action, reward, next_state, done
    
    def __len__(self):
        """
        返回缓冲区中当前存储的经验数量
        """
        return len(self.buffer)


class OUNoise:
    """
    Ornstein-Uhlenbeck过程，用于在连续动作空间中进行探索
    这种噪声具有时间相关性，适合物理系统的连续控制任务
    """
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu           # 均值
        self.theta = theta     # 回归速度参数
        self.sigma = sigma     # 波动率参数
        self.state = np.ones(self.action_dim) * self.mu  # 初始化状态
        self.reset()
    
    def reset(self):
        """
        重置噪声过程
        """
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        """
        采样一个噪声值
        """
        # Ornstein-Uhlenbeck过程的离散形式
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


class DDPG:
    """
    DDPG算法的主类
    包含Actor、Critic网络以及它们的目标网络
    """
    def __init__(self, state_dim, action_dim, max_action, device, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.001):
        # 设备选择：GPU如果可用，否则CPU
        self.device = device
        
        # 超参数
        self.gamma = gamma  # 折扣因子，衡量未来奖励的重要性
        self.tau = tau      # 目标网络 软更新参数
        
        # 初始化Actor网络和目标Actor网络
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        
        # 初始化Critic网络和目标Critic网络
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        
        # 将目标网络的参数初始化为与在线网络相同
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 定义优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 初始化OU噪声，用于探索
        self.noise = OUNoise(action_dim)
    
    def select_action(self, state, add_noise=True):
        """
        根据当前状态选择动作
        add_noise参数控制是否添加探索噪声
        """
        # 兼容 gym / gymnasium 返回值：有些环境会返回 (obs, info)
        if isinstance(state, tuple) or isinstance(state, list):
            # 取第一个元素作为观测
            state = state[0]

        # 确保是 numpy 数组（有时是列表），再转换为 torch 张量并移动到设备
        state = np.array(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # 使用Actor网络预测动作（不需要计算梯度，因为这是选择动作不是训练），只是推理，不需要反向传播
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        
        # 如果需要，添加OU噪声进行探索
        if add_noise:
            noise = self.noise.sample() # 采样噪声
            action = np.clip(action + noise, -self.actor.max_action, self.actor.max_action) # 添加噪声并裁剪动作值
        
        return action
    
    def train(self, replay_buffer, batch_size=64):
        """
        训练DDPG算法
        从经验回放缓冲区中采样一个批量，然后更新Actor和Critic网络
        """
        # 从经验回放缓冲区中采样
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # 将numpy数组转换为torch张量，并移动到相应设备
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # 计算目标Q值
        with torch.no_grad():
            # 使用目标Actor网络选择下一个动作
            next_action = self.actor_target(next_state)
            
            # 使用目标Critic网络计算下一个状态-动作对的Q值
            target_Q = self.critic_target(next_state, next_action)
            
            # 计算目标Q值：如果episode结束，只有即时奖励；否则包括未来折扣奖励
            target_Q = reward + ((1 - done) * self.gamma * target_Q)
        
        # 更新Critic网络
        # 计算当前Q值
        current_Q = self.critic(state, action)
        
        # Critic的损失函数：当前Q值和目标Q值之间的均方误差
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        # 清空Critic网络的梯度
        self.critic_optimizer.zero_grad()
        
        # 反向传播计算梯度
        critic_loss.backward()
        
        # 更新Critic网络参数
        self.critic_optimizer.step()
        
        # 更新Actor网络
        # Actor的损失函数：Critic网络对当前状态和Actor选择动作的Q值取负（因为我们想最大化Q值）
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # 清空Actor网络的梯度
        self.actor_optimizer.zero_grad()
        
        # 反向传播计算梯度
        actor_loss.backward()
        
        # 更新Actor网络参数
        self.actor_optimizer.step()
        
        # 软更新目标网络
        # 使用加权平均的方式更新目标网络参数，提高学习稳定性
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, filename):
        """
        保存模型参数
        """
        torch.save({
            'actor': self.actor.state_dict(),   # 保存Actor网络参数
            'critic': self.critic.state_dict(), # 保存Critic网络参数
            'actor_target': self.actor_target.state_dict(),   # 保存目标Actor网络参数
            'critic_target': self.critic_target.state_dict(), # 保存目标Critic网络参数
            'actor_optimizer': self.actor_optimizer.state_dict(), # 保存Actor优化器参数
            'critic_optimizer': self.critic_optimizer.state_dict(), # 保存Critic优化器参数
        }, filename)
    
    def load(self, filename):
        """
        加载模型参数
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device) # 加载模型到指定设备
            self.actor.load_state_dict(checkpoint['actor'])     # 加载Actor网络参数
            self.critic.load_state_dict(checkpoint['critic'])   # 加载Critic网络参数
            self.actor_target.load_state_dict(checkpoint['actor_target'])         # 加载目标Actor网络参数
            self.critic_target.load_state_dict(checkpoint['critic_target'])       # 加载目标Critic网络参数
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])   # 加载Actor优化器参数
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer']) # 加载Critic优化器参数
            print("模型加载成功！")
        else:
            print("模型文件不存在！")


def train_ddpg(env, agent, max_episodes=MAX_Train_EPISODES, max_steps=MAX_Train_STEPS, batch_size=64, save_interval=50):
    """
    训练DDPG算法的主函数
    """
    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(100000)
    
    # 记录训练过程中的奖励和损失
    rewards = []
    actor_losses = []
    critic_losses = []
    
    print("开始训练DDPG算法...")
    
    # 训练循环
    for episode in range(max_episodes):
        # 重置环境，获取初始状态
        reset_ret = env.reset()
        # gymnasium 的 reset 可能返回 (obs, info)
        if isinstance(reset_ret, tuple) or isinstance(reset_ret, list):
            state = reset_ret[0]
        else:
            state = reset_ret
        agent.noise.reset()  # 重置OU噪声

        episode_reward = 0          # 当前episode的累计奖励
        episode_actor_loss = 0      # 当前episode的Actor损失
        episode_critic_loss = 0     # 当前episode的Critic损失
        update_count = 0            # 本episode训练中，更新 actor网络 和 Critic网络 的次数
        
        # 单个episode的循环
        for step in range(max_steps):
            # 选择动作（添加探索噪声）
            action = agent.select_action(state)

            # 执行动作，与环境交互
            step_ret = env.step(action) # 由gym封装的step函数
            # gymnasium 的 step 返回 (obs, reward, terminated, truncated, info)
            if len(step_ret) == 5:
                next_state, reward, terminated, truncated, info = step_ret
                done = terminated or truncated
            else:
                # 兼容老的 gym API
                next_state, reward, done, info = step_ret
            
            # 将经验存储到回放缓冲区（保存原始观测，不保存 info）
            replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            
            # 如果回放缓冲区中有足够多的经验，就开始训练
            if len(replay_buffer) > batch_size:
                actor_loss, critic_loss = agent.train(replay_buffer, batch_size)
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                update_count += 1
            
            # 如果episode结束，跳出循环
            if done:
                break
        
        # 计算平均损失
        if update_count > 0:
            episode_actor_loss /= update_count
            episode_critic_loss /= update_count
        
        # 记录奖励和损失
        rewards.append(episode_reward)
        actor_losses.append(episode_actor_loss)
        critic_losses.append(episode_critic_loss)
        
        # 每50个episode输出一次训练信息
        if episode % 10 == 0:
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Actor Loss: {episode_actor_loss:.4f}, Critic Loss: {episode_critic_loss:.4f}")
        
        # 定期保存模型
        if episode % save_interval == 0:
            agent.save(f"DDPG/ddpg_model_episode_{episode}.pth")
    
    # 训练结束后保存最终模型
    agent.save("DDPG/ddpg_final_model.pth")
    print("训练完成！最终模型已保存。")
    
    return rewards, actor_losses, critic_losses


def test_ddpg(env, agent, episodes=10, render=True):
    """
    测试训练好的DDPG模型
    """
    print("开始测试模型...")
    
    test_rewards = []
    
    for episode in range(episodes):
        reset_ret = env.reset()  # 初始化环境
        if isinstance(reset_ret, tuple) or isinstance(reset_ret, list): # 兼容 gymnasium 的 reset
            state = reset_ret[0]
        else:
            state = reset_ret
        episode_reward = 0

        for step in range(Test_STEPS):  # 测试的最大步数
            # 选择动作（测试时不添加噪声）
            action = agent.select_action(state, add_noise=False)
            
            # 执行动作，兼容 gym 和 gymnasium
            step_ret = env.step(action)
            if len(step_ret) == 5:
                next_state, reward, terminated, truncated, info = step_ret
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_ret
            
            # 更新状态和累计奖励
            state = next_state
            episode_reward += reward
            
            # 如果需要渲染，显示环境
            if render:
                env.render()
            
            # 如果episode结束，跳出循环
            if done:
                break
        
        test_rewards.append(episode_reward)
        print(f"测试 Episode {episode + 1}: 奖励 = {episode_reward:.2f}")
    
    # 计算平均测试奖励
    avg_reward = np.mean(test_rewards)
    print(f"平均测试奖励: {avg_reward:.2f}")
    
    return test_rewards


def plot_training_results(rewards, actor_losses, critic_losses, window=50):
    """
    可视化训练结果
    """
    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # 绘制奖励曲线
    ax1.plot(rewards, alpha=0.5, label='Reward per Episode')
    
    # 计算移动平均奖励，使曲线更平滑
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, label=f'{window} Episode Moving Average', color='red')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward over Episodes')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制Actor损失曲线（只绘制有训练更新的回合）
    if any(actor_losses):
        valid_actor_losses = [loss for loss in actor_losses if loss > 0]
        ax2.plot(range(len(valid_actor_losses)), valid_actor_losses)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss of Actor Network over Episodes')
        ax2.grid(True)
    
    # 绘制Critic损失曲线（只绘制有训练更新的回合）
    if any(critic_losses):
        valid_critic_losses = [loss for loss in critic_losses if loss > 0]
        ax3.plot(range(len(valid_critic_losses)), valid_critic_losses)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title('Loss of Critic Network over Episodes')
        ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('DDPG/ddpg_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    主函数：整合训练、测试和可视化
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env = gym.make('Pendulum-v1')
    
    # 获取环境的状态和动作维度
    state_dim = env.observation_space.shape[0] # 状态维度
    action_dim = env.action_space.shape[0]     # 动作维度
    max_action = float(env.action_space.high[0]) # 动作的最大值
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}, 最大动作值: {max_action}")
    
    # 初始化DDPG智能体
    agent = DDPG(state_dim, action_dim, max_action, device)
    
    # 训练模型
    rewards, actor_losses, critic_losses = train_ddpg(env, agent)
    
    # 可视化训练结果
    plot_training_results(rewards, actor_losses, critic_losses)
    
    # 关闭环境（训练时不需要渲染）
    env.close()
    
    # 重新创建环境用于测试（测试时需要渲染）
    env = gym.make('Pendulum-v1')
    
    # 加载最佳模型进行测试
    agent.load("DDPG/ddpg_final_model.pth")
    
    # 测试模型
    test_rewards = test_ddpg(env, agent, episodes=Test_EPISODES, render=True)
    print(f"测试阶段的平均奖励: {np.mean(test_rewards):.2f}")
    
    # 关闭环境
    env.close()
    
    print("程序执行完毕！")


if __name__ == "__main__":
    main()