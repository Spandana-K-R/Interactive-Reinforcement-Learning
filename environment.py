import wiper_interface

wiper_system_interface_obj = wiper_interface.Wiper_System_Interface()

class Environment(gym.Env):
    
    def __init__(self, wiper_system_interface_obj, L=0.05, frac_same_action=0.1):
        
        self.wiper_system_interface_obj = wiper_system_interface_obj
        
        self.action_space           = self.wiper_system_interface_obj.action_space
        self.observation_space      = self.wiper_system_interface_obj.observation_space
        
        self.n_actions              = self.action_space.n
        self.n_features             = self.observation_space.shape[0]
        self.length_df              = len(wiper_system_interface_obj.dataset_obj)

        self.current_iter_till_done = -1
        
        self.train_rewards_list     = []
        self.max_rewards_list       = []
        self.user_agent_same_action = []
                
        self.script_L               = L
        self.frac_same_action       = frac_same_action
        self.take_advice_bool       = False
        self.finish                 = False
        
    def reset(self):
        self.current_iter_till_done = 0 
        return self.wiper_system_interface_obj.reset()
    
    def seed(self, seed=None):
        return
    
    def step(self, action):
        self.current_iter_till_done  += 1
        user_input = self.wiper_system_interface_obj.get_user_input()
        
        self.user_agent_same_action.append(int(action == user_input))
        self.frac = sum(self.user_agent_same_action)/len(self.user_agent_same_action)
        
        self.wiper_system_interface_obj.act()
        state  = self.wiper_system_interface_obj.get_current_state()
        
        reward = self.calc_reward(action, user_input)
        
        if self.frac < self.frac_same_action or np.random.random() < self.script_L:
            reward = 0.2
            self.take_advice_bool = True            
                
        self.train_rewards_list.append(reward)
        max_reward = self.calc_reward(user_input, user_input)
        self.max_rewards_list.append(max_reward)
        
        info                   = {}
        info["RL_action"]      = action
        info["correct_action"] = user_input
        info["max_reward"]     = max_reward

        mean_frac_rewards = np.mean([p/q for (p,q) in zip(self.train_rewards_list[-600:],self.max_rewards_list[-600:])])
        
        if mean_frac_rewards > 0.85 and self.current_iter_till_done > 600: #not self.script_L == 1 and
            done = True
            print("mean frac rewards is above 0.9")
            self.finish = True
        else: 
            done   = self.wiper_system_interface_obj.is_done()
        
        return state, reward, done, info
    
    def calc_reward(self, agent_input, user_input):
        return self.wiper_system_interface_obj.reward_structure[int(user_input==agent_input)][int(user_input)]
