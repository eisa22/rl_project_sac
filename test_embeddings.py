"""
Quick Test: Verify Task Embeddings Implementation

This script runs a short training (1000 steps) to verify that:
1. Environment returns correct observation dimensions
2. Agent can process task_ids correctly
3. Embeddings are being trained
4. No errors in the forward/backward pass

Usage:
    python test_embeddings.py
"""

import numpy as np
import torch
from sac_agent_embeddings import SACAgentEmbedding, PerTaskReplayBuffer
from train_metaworld_embeddings import MetaWorldMT10EnvEmbedding


def test_environment():
    """Test that environment returns correct dimensions."""
    print("="*70)
    print("TEST 1: Environment Setup")
    print("="*70)
    
    env = MetaWorldMT10EnvEmbedding(seed=42, max_episode_steps=50)
    
    obs, info = env.reset()
    
    print(f"✓ Observation shape: {obs.shape}")
    print(f"  Expected: (39,) - Pure state, NO one-hot")
    assert obs.shape == (39,), f"Expected obs shape (39,), got {obs.shape}"
    
    print(f"✓ Task ID in info: {info['task_id']}")
    print(f"  Task name: {info['task_name']}")
    assert 'task_id' in info, "Missing task_id in info dict!"
    assert 0 <= info['task_id'] < env.num_tasks, f"Invalid task_id: {info['task_id']}"
    
    # Test step
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"✓ Next observation shape: {next_obs.shape}")
    assert next_obs.shape == (39,), f"Expected next_obs shape (39,), got {next_obs.shape}"
    
    print(f"✓ Task ID consistent: {info['task_id']}")
    
    env.close()
    print("\n✅ Environment test PASSED!\n")


def test_agent_initialization():
    """Test that agent initializes correctly with embeddings."""
    print("="*70)
    print("TEST 2: Agent Initialization")
    print("="*70)
    
    obs_dim = 39  # Pure observation
    act_dim = 4
    act_limit = 1.0
    num_tasks = 10
    embedding_dim = 16
    
    agent = SACAgentEmbedding(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        num_tasks=num_tasks,
        embedding_dim=embedding_dim,
        buffer_size_per_task=1000,
    )
    
    print(f"✓ Actor embedding shape: {agent.actor.task_embedding.weight.shape}")
    assert agent.actor.task_embedding.weight.shape == (num_tasks, embedding_dim)
    
    print(f"✓ Q1 embedding shape: {agent.q1.task_embedding.weight.shape}")
    assert agent.q1.task_embedding.weight.shape == (num_tasks, embedding_dim)
    
    print(f"✓ Q2 embedding shape: {agent.q2.task_embedding.weight.shape}")
    assert agent.q2.task_embedding.weight.shape == (num_tasks, embedding_dim)
    
    print(f"✓ Replay buffer initialized for {num_tasks} tasks")
    assert len(agent.replay_buffer.buffers) == num_tasks
    
    print("\n✅ Agent initialization test PASSED!\n")
    return agent


def test_forward_pass(agent):
    """Test that forward pass works correctly."""
    print("="*70)
    print("TEST 3: Forward Pass")
    print("="*70)
    
    # Create dummy observation and task_id
    obs = np.random.randn(39).astype(np.float32)
    task_id = 3
    
    # Test actor
    print("Testing actor.act()...")
    action = agent.act(obs, task_id, deterministic=False)
    print(f"✓ Action shape: {action.shape}")
    assert action.shape == (4,), f"Expected action shape (4,), got {action.shape}"
    
    # Test with batch
    print("\nTesting batch forward pass...")
    # Important: Put tensors on same device as agent (CPU or CUDA)
    device = next(agent.actor.parameters()).device
    obs_batch = torch.randn(32, 39).to(device)
    task_ids = torch.randint(0, 10, (32,)).to(device)
    
    mu, std = agent.actor.forward(obs_batch, task_ids)
    print(f"✓ Mu shape: {mu.shape}")
    print(f"✓ Std shape: {std.shape}")
    assert mu.shape == (32, 4)
    assert std.shape == (32, 4)
    
    # Test critic
    action_batch = torch.randn(32, 4).to(device)
    q_values = agent.q1(obs_batch, action_batch, task_ids)
    print(f"✓ Q-values shape: {q_values.shape}")
    assert q_values.shape == (32,)
    
    print("\n✅ Forward pass test PASSED!\n")


def test_experience_collection(agent):
    """Test that experience collection works."""
    print("="*70)
    print("TEST 4: Experience Collection")
    print("="*70)
    
    # Add some experiences
    for task_id in range(10):
        for _ in range(10):
            obs = np.random.randn(39).astype(np.float32)
            action = np.random.randn(4).astype(np.float32)
            reward = np.random.randn()
            next_obs = np.random.randn(39).astype(np.float32)
            done = False
            
            agent.add_experience(obs, action, reward, next_obs, done, task_id)
    
    print(f"✓ Total experiences: {len(agent.replay_buffer)}")
    assert len(agent.replay_buffer) == 100  # 10 tasks × 10 experiences
    
    # Test batch sampling
    print("\nTesting batch sampling...")
    batch = agent.replay_buffer.sample_batch(batch_size=50)
    
    print(f"✓ Batch obs shape: {batch['obs'].shape}")
    print(f"✓ Batch task_ids shape: {batch['task_ids'].shape}")
    print(f"✓ Unique tasks in batch: {torch.unique(batch['task_ids']).tolist()}")
    
    assert 'task_ids' in batch, "Missing task_ids in batch!"
    assert batch['obs'].shape == (50, 39), f"Expected (50, 39), got {batch['obs'].shape}"
    assert batch['task_ids'].shape == (50,), f"Expected (50,), got {batch['task_ids'].shape}"
    
    print("\n✅ Experience collection test PASSED!\n")


def test_update(agent):
    """Test that update step works without errors."""
    print("="*70)
    print("TEST 5: Agent Update")
    print("="*70)
    
    # Fill buffer with enough experiences
    print("Filling replay buffer...")
    for task_id in range(10):
        for _ in range(100):
            obs = np.random.randn(39).astype(np.float32)
            action = np.random.randn(4).astype(np.float32) * 0.1
            reward = np.random.randn()
            next_obs = np.random.randn(39).astype(np.float32)
            done = False
            
            agent.add_experience(obs, action, reward, next_obs, done, task_id)
    
    print(f"✓ Buffer size: {len(agent.replay_buffer)}")
    
    # Get initial embedding values
    initial_actor_emb = agent.actor.task_embedding.weight.clone().detach()
    
    # Perform update
    print("\nPerforming update step...")
    losses = agent.update(batch_size=256)
    
    print(f"✓ Q1 loss: {losses['q1_loss']:.4f}")
    print(f"✓ Q2 loss: {losses['q2_loss']:.4f}")
    print(f"✓ Actor loss: {losses['actor_loss']:.4f}")
    print(f"✓ Alpha: {losses['alpha']:.4f}")
    
    # Check that embeddings changed
    updated_actor_emb = agent.actor.task_embedding.weight.clone().detach()
    embedding_change = torch.abs(updated_actor_emb - initial_actor_emb).sum().item()
    
    print(f"\n✓ Embedding parameters changed: {embedding_change:.6f}")
    assert embedding_change > 0, "Embeddings didn't change after update!"
    
    print("\n✅ Agent update test PASSED!\n")


def test_short_training():
    """Run a very short training loop to verify everything works together."""
    print("="*70)
    print("TEST 6: Short Training Loop (100 steps)")
    print("="*70)
    
    env = MetaWorldMT10EnvEmbedding(seed=42, max_episode_steps=50)
    
    agent = SACAgentEmbedding(
        obs_dim=39,
        act_dim=4,
        act_limit=1.0,
        num_tasks=10,
        embedding_dim=16,
        buffer_size_per_task=1000,
    )
    
    obs, info = env.reset()
    task_id = info['task_id']
    
    for step in range(100):
        # Get action
        if step < 20:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, task_id, deterministic=False)
        
        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_task_id = info['task_id']
        
        # Add experience
        agent.add_experience(obs, action, reward, next_obs, done, task_id)
        
        obs = next_obs
        task_id = next_task_id
        
        # Update
        if step >= 50 and step % 10 == 0:
            losses = agent.update(batch_size=32)
            if step % 30 == 0:
                print(f"  Step {step}: Q1 Loss={losses['q1_loss']:.4f}, "
                      f"Actor Loss={losses['actor_loss']:.4f}")
        
        # Reset if done
        if done:
            obs, info = env.reset()
            task_id = info['task_id']
    
    env.close()
    print("\n✅ Short training test PASSED!\n")


def main():
    print("\n" + "="*70)
    print("TASK EMBEDDINGS IMPLEMENTATION TEST SUITE")
    print("="*70 + "\n")
    
    try:
        # Run all tests
        test_environment()
        agent = test_agent_initialization()
        test_forward_pass(agent)
        test_experience_collection(agent)
        test_update(agent)
        test_short_training()
        
        # Summary
        print("="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        print("\nYour task embeddings implementation is working correctly!")
        print("\nYou can now run full training with:")
        print("  python train_metaworld_embeddings.py --run_name my_first_run")
        print("\nRecommended embedding dimensions to try:")
        print("  --embedding_dim 8   (minimal)")
        print("  --embedding_dim 16  (default, good balance)")
        print("  --embedding_dim 32  (more capacity)")
        print("  --embedding_dim 64  (maximum)")
        print("\nAfter training, analyze embeddings with:")
        print("  python analyze_embeddings.py --model_path ./models_mt10_embeddings/*/final_model.pt")
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == '__main__':
    main()
