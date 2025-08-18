#!/usr/bin/env python3
"""
Advanced DQN 28 bot (official rules)
- Card ranking: J>9>10>A>K>Q>8>7
- Points: J=3, 9=2, 10=1, A=1, K/Q/8/7=0
- Legal moves enforced
- Rich state encoding
- Verbose demo output
"""

import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Game constants
# -------------------------
SUITS = ["Hearts","Diamonds","Clubs","Spades"]
RANKS = [14,13,12,11,10,9,8,7]  # A,K,Q,J,10,9,8,7

# Mapping numeric -> card faces
CARD_RANK = {14:'A',13:'K',12:'Q',11:'J',10:'10',9:'9',8:'8',7:'7'}

# Official 28 points
CARD_POINTS = {'J':3,'9':2,'10':1,'A':1,'K':0,'Q':0,'8':0,'7':0}

# Official 28 card order (for trick winning)
CARD_ORDER = {'J':7,'9':6,'10':5,'A':4,'K':3,'Q':2,'8':1,'7':0}

# -------------------------
# Environment
# -------------------------
class TwentyEightEnv:
    def __init__(self, seed=None):
        self.players=[0,1,2,3]
        self.trump=None
        self.hands={}
        self.scores=[0,0]
        self.current_trick=[]
        self.leader=0
        self.current_player=0
        self.done=False
        if seed: random.seed(seed); np.random.seed(seed)
    
    def reset(self):
        deck=[(r,s) for s in SUITS for r in RANKS]
        random.shuffle(deck)
        self.hands={p:deck[p*8:(p+1)*8] for p in self.players}
        self.trump=random.choice(SUITS)
        self.scores=[0,0]
        self.current_trick=[]
        self.leader=0
        self.current_player=0
        self.done=False
        return self.get_state()
    
    def legal_moves(self,player):
        hand=self.hands.get(player,[])
        if not hand: return []
        if not self.current_trick: return list(range(len(hand)))
        lead_suit=self.current_trick[0][0][1]
        same_suit=[i for i,c in enumerate(hand) if c[1]==lead_suit]
        return same_suit if same_suit else list(range(len(hand)))
    
    def play_card(self,player,idx):
        card=self.hands[player].pop(idx)
        self.current_trick.append((card,player))
        return card
    
    def trick_winner_and_points(self):
        winning_card, winner = self.current_trick[0]
        for card,p in self.current_trick[1:]:
            c_face = CARD_RANK[card[0]]
            w_face = CARD_RANK[winning_card[0]]
            c_rank = CARD_ORDER[c_face]
            w_rank = CARD_ORDER[w_face]

            # same suit, higher order wins
            if card[1]==winning_card[1] and c_rank>w_rank:
                winning_card,winner=card,p
            # trump beats non-trump
            elif card[1]==self.trump and winning_card[1]!=self.trump:
                winning_card,winner=card,p

        trick_points=sum(CARD_POINTS[CARD_RANK[c[0]]] for c,_ in self.current_trick)
        return winner,trick_points
    
    def advance_after_trick(self,winner,points):
        if winner%2==0: self.scores[0]+=points
        else: self.scores[1]+=points
        self.current_trick=[]
        self.leader=winner
        self.current_player=winner
    
    def step(self,action_idx):
        player=self.current_player
        legal=self.legal_moves(player)
        if not legal: return self.get_state(),0,self.done
        if action_idx not in legal: action_idx=random.choice(legal)
        card=self.play_card(player,action_idx)
        reward=0
        if len(self.current_trick)==4:
            winner,pts=self.trick_winner_and_points()
            self.advance_after_trick(winner,pts)
            reward=pts if player==0 else -pts
        else:
            self.current_player=(self.current_player+1)%4
        if all(len(self.hands[p])==0 for p in self.players):
            self.done=True
            reward += self.scores[0]-self.scores[1]
        return self.get_state(),reward,self.done
    
    def get_state(self):
        # State encoding
        # Hand (32) + trump (4) + current trick (32) + remaining high cards (32) + scores (2) = 102
        state=np.zeros(32+4+32+32+2,dtype=np.float32)
        # AI hand
        for c in self.hands.get(0,[]):
            idx=RANKS.index(c[0])+SUITS.index(c[1])*8
            state[idx]=1.0
        # Trump
        state[32+SUITS.index(self.trump)]=1.0
        # Current trick
        for i,(c,_) in enumerate(self.current_trick):
            idx=RANKS.index(c[0])+SUITS.index(c[1])*8
            state[36+i*8+idx%8]=1.0
        # Remaining high cards
        remaining_cards={(r,s) for s in SUITS for r in RANKS}
        for hand in self.hands.values():
            remaining_cards -= set(hand)
        for c in remaining_cards:
            if CARD_ORDER[CARD_RANK[c[0]]]>=5:  # high cards: J,9,10,A
                idx=RANKS.index(c[0])+SUITS.index(c[1])*8
                state[36+32+idx]=1.0
        # Scores
        state[-2]=self.scores[0]
        state[-1]=self.scores[1]
        return state
    
    def render(self):
        print("Trump:",self.trump)
        for p in self.players:
            print(f"P{p} hand:", " ".join(f"{CARD_RANK[c[0]]}-{c[1][0]}" for c in self.hands.get(p,[])))
        if self.current_trick:
            print("Trick:",[(f"{CARD_RANK[c[0]]}-{c[1][0]}",p) for c,p in self.current_trick])
        print("Scores:",self.scores)
        print("-"*40)

# -------------------------
# Opponent
# -------------------------
def random_opponent(env,player):
    legal=env.legal_moves(player)
    return random.choice(legal) if legal else None

# -------------------------
# DQN
# -------------------------
class DQN(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,output_dim)
        )
    def forward(self,x): return self.net(x)

# -------------------------
# Replay buffer
# -------------------------
class ReplayBuffer:
    def __init__(self,size=20000):
        self.buffer=deque(maxlen=size)
    def push(self,s,a,r,ns,d): self.buffer.append((s,a,r,ns,d))
    def sample(self,batch_size):
        batch=random.sample(self.buffer,batch_size)
        s,a,r,ns,d=zip(*batch)
        return np.array(s),np.array(a),np.array(r),np.array(ns),np.array(d)
    def __len__(self): return len(self.buffer)

# -------------------------
# Training
# -------------------------
def train_dqn(episodes=8000):
    env=TwentyEightEnv()
    env.reset()
    input_dim=len(env.get_state())
    output_dim=8
    policy_net=DQN(input_dim,output_dim)
    target_net=DQN(input_dim,output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer=optim.Adam(policy_net.parameters(),lr=1e-3)
    buffer=ReplayBuffer(20000)
    gamma=0.95
    epsilon=1.0
    epsilon_min=0.05
    batch_size=64
    target_update=100
    
    for ep in range(episodes):
        state=env.reset()
        done=False
        while not done:
            legal=env.legal_moves(0)
            if not legal:
                state,_ ,done=env.step(random.choice(range(8)))
                continue
            if random.random()<epsilon:
                action=random.choice(legal)
            else:
                with torch.no_grad():
                    qvals=policy_net(torch.tensor(state,dtype=torch.float32))
                    qvals_np=qvals.numpy()
                    qvals_filtered=[(i,qvals_np[i]) for i in legal]
                    action=max(qvals_filtered,key=lambda x:x[1])[0]
            next_state,reward,done=env.step(action)
            buffer.push(state,action,reward,next_state,done)
            state=next_state
            
            # Opponents
            while not done and env.current_player!=0:
                player=env.current_player
                act=random_opponent(env,player)
                if act is None: break
                state,_ ,done=env.step(act)
        
            if len(buffer)>=batch_size:
                s,a,r,ns,d=buffer.sample(batch_size)
                s=torch.tensor(s,dtype=torch.float32)
                a=torch.tensor(a,dtype=torch.long)
                r=torch.tensor(r,dtype=torch.float32)
                ns=torch.tensor(ns,dtype=torch.float32)
                d=torch.tensor(d,dtype=torch.float32)
                qvals=policy_net(s)
                qval=qvals.gather(1,a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    qnext=target_net(ns).max(1)[0]
                    qtarget=r+gamma*qnext*(1-d)
                loss=nn.MSELoss()(qval,qtarget)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epsilon=max(epsilon*0.995,epsilon_min)
        if ep%500==0: print(f"Episode {ep} complete")
        if ep%target_update==0:
            target_net.load_state_dict(policy_net.state_dict())
    return policy_net, env

# -------------------------
# Demo
# -------------------------
# -------------------------
# Demo with full round debug
# -------------------------
def demo_verbose(net, env, games=3):
    for g in range(games):
        state = env.reset()
        print(f"\n=== Game {g+1} ===")
        print(f"Trump: {env.trump}\n")
        round_num = 1
        while not env.done:
            trick_cards = []
            for i in range(4):
                player = env.current_player
                legal = env.legal_moves(player)
                if player == 0:
                    # AI chooses
                    with torch.no_grad():
                        qvals = net(torch.tensor(state, dtype=torch.float32))
                        qvals_np = qvals.numpy()
                        qvals_filtered = [(i, qvals_np[i]) for i in legal]
                        action = max(qvals_filtered, key=lambda x: x[1])[0]
                else:
                    # Opponent random
                    action = random_opponent(env, player)
                card_played = env.hands[player][action]
                state, reward, done = env.step(action)
                trick_cards.append((player, CARD_RANK[card_played[0]], card_played[1]))
            
            # Determine winner and trick points
            winner, trick_points = env.trick_winner_and_points()
            env.advance_after_trick(winner, trick_points)

            # Print verbose info
            print(f"--- Round {round_num} ---")
            for p, rank, suit in trick_cards:
                print(f"Player {p+1} played: {rank}-{suit[0]}")
            print(f"Winner: Player {winner+1}")
            print(f"Total points added: {trick_points}")
            print("------------------------\n")
            round_num += 1

        # Final scores
        print(f"Game {g+1} complete. Final Scores: Team 1: {env.scores[0]} | Team 2: {env.scores[1]}")

# -------------------------
# Run
# -------------------------
if __name__=="__main__":
    net,env=train_dqn(episodes=500)
    demo_verbose(net,env,3)
