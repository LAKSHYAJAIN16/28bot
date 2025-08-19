import time
import mcts
print('HAS_TORCH:', mcts.HAS_TORCH)
if not mcts.HAS_TORCH:
    print('PyTorch not available; aborting training.')
else:
    total_cycles = 100          # increase for longer training
    episodes_per_cycle = 10    # self-play games per cycle
    base_iterations = 400      # MCTS iters to start
    for cycle in range(1, total_cycles+1):
        iters = base_iterations + (cycle//10)*100  # slowly ramp iterations
        print(f"\n=== Cycle {cycle}/{total_cycles} | episodes={episodes_per_cycle} | iters={iters} ===")
        mcts.train_policy_value(
            episodes=episodes_per_cycle,
            iterations=iters,
            batch_size=256,
            epochs=8,
            lr=1e-3
        )
        ckpt = f"pv_cycle_{cycle:03d}.pt"
        mcts.NN_EVAL.save(ckpt)
        print(f"Saved checkpoint: {ckpt}")
        # quick sanity check game (low iterations)
        try:
            from mcts import play_games
            play_games(num_games=1, iterations=150)
        except Exception as e:
            print('Play sanity-check failed:', e)
        time.sleep(2)  # small pause to keep things smooth
    mcts.NN_EVAL.save('pv_final.pt')
    print('Saved final weights to pv_final.pt')