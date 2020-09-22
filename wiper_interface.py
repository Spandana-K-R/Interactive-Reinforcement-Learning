import user
import dataset

user_obj    = user.User()
dataset_obj = dataset.Wiper_System_Dataset()

class Wiper_System_Interface():
    
    def __init__(self, user_obj, dataset_obj):
        self.user_obj          = user_obj
        self.dataset_obj       = dataset_obj
        
        self.low               = np.array(self.dataset_obj.df_states.min(axis=0))
        self.high              = np.array(self.dataset_obj.df_states.max(axis=0))
        
        self.action_value      = self.dataset_obj.df_actions.unique()
        
        self.action_space      = spaces.Discrete(len(self.action_value))
        self.observation_space = spaces.Box(low=self.low, high=self.high)
        
        self.create_reward_structure()
        
        self.current_i         = 0
    
    def current_index(self):
        if self.current_i < len(self.dataset_obj):
            return self.current_i
        else:
            self.current_i = 0
            return self.current_i
    
    def get_current_state(self):
        return self.dataset_obj[self.current_index()][0]
    
    def reset(self):
        self.current_i = 0
        return self.get_current_state()
    
    def get_user_input(self):
        return self.user_obj.user_action(self.current_index())
    
    def create_reward_structure(self):
        input_occurences      = list(self.dataset_obj.df_actions.value_counts(sort=False))
        total_inputs          = sum(input_occurences)
        self.reward_structure = []
        positive_rewards      = []
        negative_rewards      = []
        for input_occ in input_occurences:
            input_reward      = 1 - (input_occ/total_inputs) #2/(len(input_occurences))
            positive_rewards.append(input_reward)
            negative_rewards.append(0 - input_reward)
            
        self.reward_structure.append(negative_rewards)
        self.reward_structure.append(positive_rewards)
    
    def is_done(self):
        return (self.current_index()+1) == len(self.dataset_obj)
    
    def act(self):
        self.current_i += 1
