import dataset
import environment
import network
import dq_agent

data = dataset.Wiper_System_Dataset()
env = environment.Environment()
model = network.Net(env.n_features).to(device)
DQN = dq_agent.Deep_Q_Agent()

class Testing():
    def __init__(self, data, env, model, DQN):
        self.dataset_obj = data
        self._env = env
        self.model_obj = model
        self.dqn_obj = DQN
        
        self.iter_length = len(self.dataset_obj)
        self.test_accuracy_frac_same_action = []
        
    def run_test(self):
        self.model_obj.eval()
        action_true_list = []
        action_predict_list = []
        for idx in range(self.iter_length):
            state, true = self.dataset_obj[idx]
            predict = torch.argmax(self.dqn_obj.get_Q(state), dim=1)
            self.test_accuracy_frac_same_action.append(int(predict==true))
            action_true_list.append(true)
            action_predict_list.append(predict)
        
        self.test_accuracy = sum(self.test_accuracy_frac_same_action)/len(self.test_accuracy_frac_same_action)
        print(confusion_matrix(action_true_list,action_predict_list))
        return self.test_accuracy
    
    def plot(self, x, y):
        plt.figure()
        
        plt.show()
        return
