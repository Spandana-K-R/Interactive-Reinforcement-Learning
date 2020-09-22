import environment
import dq_agent

env = environment.Environment()
agent_obj = dq_agent.Deep_Q_Agent()


class Execute_Agent():
    def __init__(self, env, agent_obj, n_episodes=20, batch_size=25, gamma=0.75, L_decay=0.9):
        
        self._env          = env
        self.agent_obj     = agent_obj
        
        self.n_episodes    = n_episodes
        self.batch_size    = batch_size
        
        self.gamma         = gamma
        self.script_L_decay= L_decay
        
        self.train_accuracy_frac_same_action = []
        
    def train_helper(self, minibatch):
        
        Q_predict, Q_target = [], []
        for state,action,reward,new_state,done in minibatch:
            predict = self.agent_obj.get_Q(state)
            target  = self.agent_obj.get_Q(state)
            if done:
                target[0][action] = reward
            else:
                temp, _ = torch.max(self.agent_obj.get_Q(new_state), dim=1)
                target[0][action] = reward + self.gamma*temp.item()
            
            Q_predict.append(predict)#.unsqueeze(0))
            Q_target.append(target)#.unsqueeze(0))
        Q_predict = torch.cat(Q_predict, dim=0)
        Q_target = torch.cat(Q_target, dim=0).detach().requires_grad_(False)
        return self.agent_obj.train_agent(Q_predict,Q_target)

        
    def episode_run(self):
       
        for e in range(self.n_episodes):
            state                 = self.agent_obj.preprocess_state(np.array(self._env.reset()))
            done                  = False
            action                = self.agent_obj.select_action(state, self.agent_obj.get_epsilon())
            counter_take_advice   = 0
            counter_num_till_done = 0
            
            while not done:
                new_state, reward, done, info = self._env.step(action)
                
                new_state         = self.agent_obj.preprocess_state(new_state)
                new_action        = self.agent_obj.select_action(new_state, self.agent_obj.get_epsilon())
                
                if self._env.take_advice_bool == True:
                    counter_take_advice += 1
                    new_action = self.agent_obj.get_advice(new_state)
                    self._env.take_advice_bool = False
        
                self.agent_obj.to_remember(state, action, reward, new_state, done)                
                    
                state         = new_state
                action        = new_action
                counter_num_till_done += 1
                
                if len(self.agent_obj.memory) > self.batch_size:
                    minibatch = random.sample(self.agent_obj.memory, self.batch_size)
                    self.train_helper(minibatch)
            
            print('Episode {}, counter_advice {}, counter_done {}, reward {}, max_reward {}'.format(e,counter_take_advice,counter_num_till_done+1,sum(self._env.train_rewards_list),sum(self._env.max_rewards_list)))
            
            self._env.script_L *= self.script_L_decay
        
            self._env.user_agent_same_action = []
            self._env.train_rewards_list     = []
            self._env.max_rewards_list       = []
            
            self.train_accuracy_frac_same_action.append(self._env.frac)
            
            if self._env.finish == True:
                self._env.finish = False
                print('Solved !!! in {} episodes'.format(e+1))
                return
            
        print('Did NOT solve after {} episodes'.format(e+1))
        return(e+1)
