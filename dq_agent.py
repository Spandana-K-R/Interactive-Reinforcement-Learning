import environment
import network

env = environment.Environment()
model_obj = network.Net(env.n_features).to(device)

#gamma    = discount factor in the Bellman eqs
#epsilon  = related to exploration
#alpha    = Adam - learning rate and its decay is alpha and alpha decay
#script L = for getting advice 
#script C = for getting good advice within the get_advice function


class Deep_Q_Agent():
    def __init__(self, env, model_obj, epsilon=0.4, epsilon_min=0.01,
                 epsilon_log_decay=0.9, alpha=0.01, C=0.75):
        self._env          = env
        self.model_obj     = model_obj
        self.memory        = deque(maxlen = 1000000)
        
        self.criterion     = nn.MSELoss()
        self.optimizer     = torch.optim.Adam(self.model_obj.parameters(), lr=alpha)
        
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        
        self.script_C      = C
        
        
    def to_remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))
    
    
    def select_action(self, state, epsilon):
        #create subset A_s --- NO IDEA!
        if np.random.random() < epsilon:
            return self._env.action_space.sample()
        else:
            self.model_obj.eval()
            return torch.argmax(self.get_Q(state),dim=1)
    
    
    def get_advice(self, state):
        #create subset A_s --- NO IDEA!
        if np.random.random() < self.script_C:
            self.model_obj.eval()
            return torch.argmax(self.get_Q(state),dim=1)
        else:
            self.model_obj.eval()
            return torch.argmin(self.get_Q(state),dim=1)
    
    
    def get_epsilon(self):
        return(max(self.epsilon_min, self.epsilon))
    
    def _to_variable(self,x):
        return torch.autograd.Variable(torch.Tensor(x))    
    
    def preprocess_state(self,state):
        return self._to_variable(state.reshape(-1,self._env.n_features))
    
    def get_Q(self,states):
        states = self.preprocess_state(states)
        self.model_obj.eval()
        return self.model_obj(states)
    
    def train_agent(self, Q_pred, Q_true):
        self.model_obj.train()
        self.optimizer.zero_grad()
        loss = self.criterion(Q_pred,Q_true)
        loss.backward()
        self.optimizer.step()
        
#         for param in self.model_obj.parameters(): print(param.data)
        
#         print("............. STOP ..............")
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
