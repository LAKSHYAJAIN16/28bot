import mcts
print('HAS_TORCH:', mcts.HAS_TORCH)
if not mcts.HAS_TORCH:
    print('PyTorch not available; aborting training.')
else:
    mcts.train_policy_value(episodes=20, iterations=200, batch_size=128, epochs=3, lr=1e-3)
    mcts.NN_EVAL.save('pv.pt')
    print('Saved weights to pv.pt')
    from mcts import play_games
    play_games(num_games=1, iterations=250)