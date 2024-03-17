import torch
import torch.nn as nn

"""
A priority-queue realization of the Dijkstra algorithm based on the following:
    - https://discuss.pytorch.org/t/dijkstra-algorithm-in-pytorch/115588/3
    - https://cp-algorithms.com/graph/dijkstra_sparse.html
"""

# Currently, this only works for single attention head
class DijkstraPQ(nn.Module):
    """Priority Q implelmentation in PyTorch

    Args:
        object ([torch.Tensor]): [The Queue to work on]
    """

    def __init__(self, reduction='none'):
        super(DijkstraPQ, self).__init__()
        self.reduction = reduction

    # def __init__(self, val):
    #     self.q = torch.tensor([[val, 0]])
    #     # self.top = self.q[0]
    #     # self.isEmpty = self.q.shape[0] == 0

    def create_q(self, val):        

        self.q = torch.tensor([[val, 0]])
        # self.top = self.q[0]
        # self.isEmpty = self.q.shape[0] == 0        

    def push(self, x):
        """Pushes x to q based on weightvalue in x. Maintains ascending order

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]
            x ([torch.Tensor]): [[index, weight] tensor to be inserted]

        Returns:
            [torch.Tensor]: [The queue tensor after correct insertion]
        """
        #if type(x) == np.ndarray:
        #    x = torch.tensor(x)
        if self.isEmpty():
            self.q = x
            self.q = torch.unsqueeze(self.q, dim=0)
            return
        idx = torch.searchsorted(self.q.T[1].contiguous(), x[1].contiguous())  # can be possibly leveraged to scale up to multiple attn heads
        #print(idx)
        self.q = torch.vstack([self.q[0:idx], x, self.q[idx:]]).contiguous()

    def top(self):
        """Returns the top element from the queue

        Returns:
            [torch.Tensor]: [top element]
        """
        return self.q[0]

    def pop(self):
        """pops(without return) the highest priority element with the minimum weight

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]

        Returns:
            [torch.Tensor]: [highest priority element]
        """
        if self.isEmpty():
            print("Can Not Pop")
        self.q = self.q[1:]

    def isEmpty(self):
        """Checks is the priority queue is empty

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]

        Returns:
            [Bool] : [Returns True is empty]
        """
        return self.q.shape[0] == 0

    def forward(self, adj):

        #n = adj.shape[0]
        n = adj.shape[-2]
        #distance_matrix = torch.zeros([n, n])
        distance_matrix = torch.zeros_like(adj)
        for dim1 in range(adj.shape[0]):
            for dim2 in range(adj.shape[1]):
                for i in range(n):
                    #u = torch.zeros(n, dtype=torch.bool)
                    d = torch.inf * torch.ones(n)
                    d[i] = 0
                    #q = DijkstraPQ(i)
                    self.create_q(i)
                    #while not q.isEmpty():
                    #print('q created!')  # delete
                    #print(f'q = {self.q}')  # delete
                    while not self.isEmpty():
                        #v, d_v = q.top()  # point and distance
                        #print(f'top = {self.top()}')  # delete                      
                        v, d_v = self.top()
                        #print(f'v = {v}')  # delete
                        #print(f'd_v = {d_v}')  # delete
                        v = v.int()
                        #q.pop()
                        self.pop()
                        if d_v != d[v]:
                            continue
                        #for j, py in enumerate(adj[v]):
                        for j, py in enumerate(adj[dim1,dim2,v]):
                            if py == 0 and j != v:
                                continue
                            else:
                                to = j
                                weight = py
                                if d[v] + py < d[to]:
                                    d[to] = d[v] + py
                                    #q.push(torch.Tensor([to, d[to]]))
                                    self.push(torch.Tensor([to, d[to]]))
                    #distance_matrix[i] = d        
                    distance_matrix[dim1,dim2,i] = d

        return distance_matrix  # can return more, add here

    # original version
    # def forward(self, adj):

    #     #n = adj.shape[0]
    #     n = adj.shape[1]
    #     distance_matrix = torch.zeros([n, n])
    #     for i in range(n):
    #         u = torch.zeros(n, dtype=torch.bool)
    #         d = torch.inf * torch.ones(n)
    #         d[i] = 0
    #         #q = DijkstraPQ(i)
    #         self.create_q(i)
    #         #while not q.isEmpty():
    #         while not self.isEmpty():
    #             #v, d_v = q.top()  # point and distance
    #             v, d_v = self.top()
    #             v = v.int()
    #             #q.pop()
    #             self.pop()
    #             if d_v != d[v]:
    #                 continue
    #             for j, py in enumerate(adj[v]):
    #                 if py == 0 and j != v:
    #                     continue
    #                 else:
    #                     to = j
    #                     weight = py
    #                     if d[v] + py < d[to]:
    #                         d[to] = d[v] + py
    #                         #q.push(torch.Tensor([to, d[to]]))
    #                         self.push(torch.Tensor([to, d[to]]))
    #         distance_matrix[i] = d        

    #     return distance_matrix  # can return more, add here