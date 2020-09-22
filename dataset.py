class Wiper_System_Dataset():
    def __init__(self, csv_path):
        self.df       = pd.read_csv(csv_path, usecols = ["Timestamp", "Speed", "Temperature", "Windspeed", 
                                                         "Winddirection", "Humidity", "Perceptration", "RainMM", 
                                                         "wiper","Timestamp2", "SMean", "WMean", "SMin", "WMin", 
                                                         "SMax", "WMax", "Timestamp3", "pi3", "sumPi3"])
        self.filter_col    = [col for col in self.df if not (col.startswith('Time') or col=='wiper')]
        self.filter_index  = [idx for idx in self.df.index if (self.df.wiper.iloc[idx] == 1 or self.df.wiper.iloc[idx] == 4)]
        self.df.drop(index = self.filter_index, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df_s          = self.df[self.filter_col]
        
        self.X_scaling     = StandardScaler()
        self.df_states     = self.X_scaling.fit_transform(self.df_s)
        
        self.df['wiper']   = self.df['wiper'].apply(lambda w: w-1 if w>0 else w)
        self.df_actions    = self.df['wiper']
        
    def __getitem__(self, index):
        self.row   = self.df_states[index,:] #self.df_states.iloc[index,:].values
        self.row   = self.row.reshape(1, self.row.shape[0])
        self.label = self.df_actions[index]
        return(self.row,self.label)
    
    def __len__(self):
        return(len(self.df_actions))
