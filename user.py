import dataset
dataset_obj = dataset.Wiper_System_Dataset()

class User():
    def __init__(self, dataset_obj):
        self._dataset_obj = dataset_obj
        
    def user_action(self, index):
        return self._dataset_obj.df_actions[index]
