
def weight_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)


class Net(nn.Module):
    def __init__(self, L_in=15, kernel_size=3, num_classes=3):
        super(Net,self).__init__()
        self.L_in        = L_in
        self.kernel_size = kernel_size

        self.fc1         = nn.Sequential(nn.Linear(self.L_in,11, bias=True), nn.PReLU())
        self.fc1.apply(weight_init)

        self.fc2         = nn.Sequential(nn.Linear(11,7, bias=True), nn.PReLU())
        self.fc2.apply(weight_init)

        self.fc3         = nn.Linear(7, num_classes)
        self.fc3.apply(weight_init)

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
