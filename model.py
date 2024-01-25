import torch        
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import graph
import coarsening
from torch.nn.parameter import Parameter
class CayleyFilter(torch.nn.Module):
    # print("  INSIDE CayleyFilter CONS")
    """The Cayley Filter model."""
    def __init__(self, order, jacobi_iterations, in_channels, out_channels): 
        super(CayleyFilter, self).__init__()
        self.order = order      
        self.jacobi_iterations = jacobi_iterations         
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h=1 # parameter
        self.real_weights = Parameter(data=torch.Tensor(self.in_channels*(order+1) , out_channels), requires_grad=True)
        self.imag_weights = Parameter(data=torch.Tensor(self.in_channels*(order+1) , out_channels), requires_grad=True)
        # initialize weights and biases
        torch.nn.init.xavier_normal_(self.real_weights, gain=1.0)
        torch.nn.init.xavier_normal_(self.imag_weights, gain=1.0)
        
# ************************  Forming Graph  ********************************
    
    def forward(self, x):
        counter=0
        # print("INSIDE CayleyFilter FORWARD counter {}". format(counter))
        counter +=1
        N = x.shape[0] 
        m = x.shape[2] 
        M = x.shape[2]*x.shape[2]
        z = graph.grid(m)
        dist, idx = graph.distance_sklearn_metrics(z)
        A = graph.adjacency(dist, idx)
        L = graph.laplacian(A, normalized=True)# <class 'scipy.sparse.csr.csr_matrix'>
        diag_L = L.diagonal()           # torch.tensor(L.diagonal())       # real number laplacian matrix
        L_off_diag = L - np.diag(diag_L) #torch.tensor(L - np.diag(diag_L))

        def compute_sparse_D_inv_indices(M):  # pixels, coarsest
            """Computes the indices required for constructing a sparse version of D^-1."""
            idx_main_diag = torch.unsqueeze(torch.arange(2*M), 1).tile(1,2)
            idx_diag_ur = torch.cat([torch.unsqueeze(torch.arange(M), 1), torch.unsqueeze(torch.arange(M)+M, 1) ], 1)
            idx_diag_ll = torch.cat([torch.unsqueeze(torch.arange(M)+M, 1), torch.unsqueeze(torch.arange(M),1)],1)
            idx = torch.cat([idx_main_diag, idx_diag_ur, idx_diag_ll], 0)
            # print("IDXXXXXXXX", idx.size()) # torch.Size([3136, 2]), ([784, 2])
            # print("sparse or not idx  Dinv", type(idx))
            return idx  # 4M x 2
        
        def compute_sparse_R_indices(L_off_diag):
            """Computes the indices required for constructing a sparse version of R."""
            idx_L = np.asarray(np.where(L_off_diag)).T 
            idx_L_sh = idx_L + np.expand_dims(np.asarray([M,M]), 0) #shifted all coordinates-> [x+m,  y+M]
            idx = np.concatenate([idx_L, idx_L_sh])                                      #axis=0 default
            # print("sparse or not idx  Rsp", type(idx))
            return idx                  #shape->(2*nnz,2) the second half of rows show shifted coordinates by M (#nodes)

        def compute_sparse_numerator_projection_indices(L,M):
            """Computes the indices required for constructing the numerator projection sparse matrix."""
            idx_L = np.asarray(np.where(L.toarray())).T   # coordinates of non-zero elements of L
            idx_L_sh = idx_L + np.expand_dims(np.asarray([M,M]), 0) # shifts idx by [M, M] nnz
            idx_diag_ur = np.concatenate([np.expand_dims(np.arange(0, M), 1), np.expand_dims(np.arange(0, M)+ M,1)], 1) #(M,2)
            idx_diag_ll = np.concatenate([np.expand_dims(np.arange(0, M) + M,1), np.expand_dims(np.arange(0, M),1)], 1) #(M,2)
            idx = np.concatenate([idx_L, idx_L_sh, idx_diag_ur, idx_diag_ll])
            # print("sparse or not idx  sparse_numerator_projection", type(idx))
            return idx                   # similar to R but for all L not only for L_off_diag      ->shape:(2nnz+2M, 2)


        def CayleyConv(diag_L, L_off_diag):
            # print("INSIDE CayleyFilter CAYLEYCONV")             
        # **************************    COMPUTES D_inv      Diag^(-1)(h*delta+iI) ************************** 
            # print("h.device", self.h.device) #cpu
            # print("diag_L.device", diag_L.device) #erroe??????????????????????            
            D_real = torch.tensor(self.h*diag_L)  # D_real =  self.h*diag_L.clone().detach() 
            D_real = D_real.squeeze(0)
            # print("D_real", D_real.shape)    # torch.Size([784])
            D = torch.complex(D_real, torch.ones_like(D_real)) 
            # print("D", D[:10])
            D_inv = torch.pow(D, -torch.ones_like(D)) 
            # print("D_inv", D_inv.shape)       # torch.Size([784])
            idx = compute_sparse_D_inv_indices(M)  
            # print("idxxxxxxxxx", idx.shape)   # torch.Size([3136, 2]) [784, 2]
            vals = torch.concat([torch.real(D_inv), torch.real(D_inv), -torch.imag(D_inv), torch.imag(D_inv)], 0)
            vals = vals.reshape(1,-1).squeeze(0)
            # print("TTTTTTTTTTTTTT", vals.size()) # torch.Size([3136]) 4M
            D_inv_ext_sp = torch.sparse_coo_tensor(idx.T, vals, [2*M, 2*M])
            D_inv_ext_sp = torch._coalesce(D_inv_ext_sp).type(torch.float32)
            # print("sparse or not D_inv_ext_sp", D_inv_ext_sp.is_sparse)
            # ************************** COMPUTES R   off(h*delta+iI) **************************
            idx = compute_sparse_R_indices(L_off_diag)
            # print("idx", idx.shape) # (12792, 2) 
            # print(L_off_diag.shape, L_off_diag.shape )       # (784, 784) (784, 784)
            vals_L = torch.squeeze(torch.from_numpy(self.h*L_off_diag[np.where(L_off_diag)]))   
            # print("vals_L", vals_L.shape, type(vals_L)) # torch.Size([6396])
            vals = torch.cat([vals_L, vals_L], 0)
            # print("vals.shape", vals.shape) #torch.Size([12792])
            R_sp = torch.sparse_coo_tensor(idx.T, vals, [M*2, M*2]).type(torch.float32)
            # print("R_sp.shape", R_sp.shape)
            R_sp = torch._coalesce(R_sp) # unique idx and inorder, canonical form
            # print("sparse or not R_sp", R_sp.is_sparse)
            # ************************** COMPUTES   h*delta-iI **************************
            idx = compute_sparse_numerator_projection_indices(L,M)  
            vals_L = torch.squeeze(torch.from_numpy(self.h*L[np.where(L.toarray())]))  
            vals = torch.cat([vals_L, vals_L, torch.ones([M,]), -torch.ones([M,])], 0) # array shape->(2nnz+2M,) 
            cayley_op_neg_sp = torch.sparse_coo_tensor(idx.T, vals, [2*M, 2*M]) # This function returns an uncoalesced tensor
            cayley_op_neg_sp = torch._coalesce(cayley_op_neg_sp).type(torch.float32)
            # print("sparse or not cayley_op_neg_sp", cayley_op_neg_sp.is_sparse)
            #*****************************   Applies Cayley and Jacobi method     ******************************
            # print(" forward method of Cayleyfilter x.shape", x.shape)#torch.Size([128, 1, 28, 28]),[128, 3, 14, 14]
            # print(" M, self.in_channels", M, self.in_channels)
            x_1 = torch.reshape(x, (N, self.in_channels, M))#x.shape[2]*x.shape[3]
            x_2 = torch.permute(x_1,(2,0,1))
            c_transform = x_2.reshape([x_2.shape[0], -1])   # M*NF
            J =  torch.sparse.mm(D_inv_ext_sp, R_sp).type(torch.float32)#.type(torch.DoubleTensor)
            y_0 = torch.cat([c_transform, torch.zeros_like(c_transform)], 0)#.type(torch.DoubleTensor) 
            last_sol = y_0.cpu()
            x_3 = torch.permute(x_2,[1,0,2]) 
            list_x_pos_exp = list(torch.unsqueeze(x_3, 0).type(torch.complex64).unsqueeze(0))
            for _ in range(self.order):   # order of polynomial
                b_j = torch.sparse.mm(D_inv_ext_sp, cayley_op_neg_sp)#.to(device) # 2M*NF
                # print("b_j.device, last_sol.device", b_j.device, last_sol.device) # cpu cuda:0
                b_j = torch.sparse.mm(b_j, last_sol)   
                k = 1
                while k <= self.jacobi_iterations:
                    y_k = b_j-torch.sparse.mm(J, last_sol)
                    k+=1    #y_K=y    2M x NF
                c_sol_complex = torch.complex(y_k[:M,:], y_k[M:, :])        # M x N*Fin
                c_sol_reshaped = torch.reshape(c_sol_complex, [M, -1, self.in_channels])   # M x N x Fin
                c_sol_reshaped = c_sol_reshaped.permute(1, 0, 2)                      # N x M x Fin
                list_x_pos_exp.append(torch.unsqueeze(c_sol_reshaped, 0))            # 1 x N x M x F     list_x_pos_exp
                last_sol = y_k  
            # print("last sol.device", last_sol.device)
            x_pos_exp = torch.concat(list_x_pos_exp, 0)      # (n_h*order+1) x N x M x Fin             put all yK together
            x_pos_exp = torch.permute(x_pos_exp, [1,2,0,3])  # N x M x n_h*(order+1) x Fin
            x_pos_exp = torch.reshape(x_pos_exp, [N*M, -1])  # N*M x n_h*(order+1)*Fin      
            real_weights = Parameter(data=torch.Tensor(self.in_channels*(self.order+1) , self.out_channels), requires_grad=True)
            torch.nn.init.xavier_normal_(real_weights, gain=1.0)
            imag_weights = Parameter(data=torch.Tensor(self.in_channels*(self.order+1) , self.out_channels), requires_grad=True)
            torch.nn.init.xavier_normal_(imag_weights, gain=1.0)
            W_pos_exp = torch.complex(real_weights, -imag_weights)
            # print("W_pos_exp", W_pos_exp.shape, W_pos_exp.sum())
            x_pos_exp_filt = torch.sparse.mm(x_pos_exp, W_pos_exp)
            x_filt = 2*torch.real(x_pos_exp_filt)
            x_filt = torch.reshape(x_filt, [N, M, self.out_channels])
            # print(x_filt.shape, x_filt.dtype, x_filt.sum()) # N, M, C  
            x_filt_2d = x_filt.reshape(N, m, m, self.out_channels)
            # x_filt_2d = x_filt_2d.to_sparse()
            # print(" sparse or not _filt_2d", x_filt_2d.is_sparse) #True
            return x_filt_2d  #torch(x_filt_2d)
        x_filtered = CayleyConv(diag_L, L_off_diag)    #?????????
        # print(" output of forward method, x_filtered.shape", x_filtered.is_sparse, x_filtered.shape)    
        return x_filtered


class CayleyNet(torch.nn.Module):
    # Declaring the Architecture
    # print("INSIDE CayleyNET CONS")
    def __init__(self, order, jacobi_iterations, in_channels, out_channels, hid_dim, learning_rate=1e-4, 
                momentum=0.9, regularization=5e-4, bias=True, activation=None): 
        super(CayleyNet, self).__init__()
        self.order = order
        self.jacobi_iterations = jacobi_iterations 
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.pool=coarsening.coarsen(4,1)
        self.h = 1
        self.hid_dim = hid_dim
        self.real_weights = Parameter(data=torch.Tensor(in_channels*(order+1) , out_channels), requires_grad=True)
        self.imag_weights = Parameter(data=torch.Tensor(in_channels*(order+1) , out_channels), requires_grad=True)
        self.bias1 = Parameter(torch.Tensor(self.hid_dim))
        self.bias2 = Parameter(torch.Tensor(out_channels))
        torch.nn.init.xavier_normal_(self.real_weights, gain=2.0)
        torch.nn.init.xavier_normal_(self.imag_weights, gain=2.0)
        self.cayley1 = CayleyFilter(order, jacobi_iterations, in_channels=1, out_channels=self.hid_dim)
        self.cayley2 = CayleyFilter(order, jacobi_iterations, in_channels=self.hid_dim, out_channels=32)  
        self.fc1 = torch.nn.Linear(32*7*7, 128*3*3) # initializes automatically nn.Linear(fin, fout)
        self.fc2 = torch.nn.Linear(128*3*3, 10)      
        self.dropout1 = torch.nn.Dropout(0.25) # Input: (N, C, H, W) or (N, C, L)  # Output: same shape as input
        self.dropout2 = torch.nn.Dropout(0.5)
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.logsoftmax = torch.nn.LogSoftmax(dim=1) #calculates softmax across the columns.
        
    # defining a computational graph       
    def forward(self, x):  
        y = self.cayley1(x).clone().detach()        # x: [128, 1, 28, 28]
        # print(" after cayley y.shape", y.shape)     # y: [128, 28, 28, 3]
        # print(" 1 bias.shape",  self.bias1.shape)   # torch.Size([3])
        # y = torch.add(self.bias1.cpu(), y)   
        y = F.leaky_relu(y)
        # print(" 1 F.relu(y).shape", F.relu(y).shape)   # y: [128, 28, 28, 3]
        y = torch.permute(y, [0,3,1,2])                # pooling wants (B x C x m X m)
        y = self.pool(y)                             
        # print(" 1 after pooling", y.shape)             # [128, 3, 14, 14]
        y = self.dropout1(y)
        # print(" shape of y after dropout", y.shape)     # [128, 3, 14, 14]
        #***************************************** second cayley **********************************************************
        y = torch.permute(y, [0,1,2,3]) 
        # print(" input to the second cayley later", y.shape) # y: [128, 3, 14, 14]
        y = self.cayley2(y)#.clone().detach() 
        # print(" self.cayley2(y).shape", y.shape)            # y: [128, 14, 14, 32]
        # print(" 2 bias.shape",  self.bias2.shape)
        # y = torch.add(y, self.bias2.cpu())   
        y = F.leaky_relu(y)
        # print(" 2 F.relu(y).shape", F.relu(y).shape) 
        y = torch.permute(y, [0,3,1,2])
        y = self.pool(y)                             
        # print(" 2 after pooling", y.shape)                  # y: [128, 32, 7, 7]   
        y = self.dropout2(y)    
        #***************************************** Linear layers ********************************************** 
        y = y.contiguous().view(y.shape[0], -1)  # flatten, y must be vectors: B x out_channels
        y = self.fc1(y)
        y = F.leaky_relu(y)
        y = self.fc2(y) # final Linear layer will produce a set of raw-score logits
                        # (unnormalized log-odds-ratios), one for each of the classes.
        # print(" y after last layer", y, y.shape, type(y), y.dtype) # torch.Size([128, 10])

        # *********       logsoftmax    *************
        # logits = self.logsoftmax(y)  #, -1      # B x num_classes
        # print(" y after logsoftmax", y.shape, type(y), y.dtype)
        # _, predicted_class = torch.max(logits, dim=1)

        # ************** integer encoding *************************
        # _, predicted_class = torch.max(y, dim = 1) #value, idx
        # print("  predicted_class", predicted_class)
        return   y   #predicted_class  #logits #predicted_label







# obj=CayleyNet(order=2, jacobi_iterations=2, in_channels=1, out_channels=32, hid_dim=3, learning_rate=1e-4, 
#                 momentum=0.9, regularization=5e-4, bias=True, activation=None)


# from loader import *
# train_loader, validation_loader, test_loader = data_loader()
# for batch_idx, (image, labels) in enumerate(train_loader):
#     image, labels= image, labels

# print("obj(img)", obj(image).shape, type(obj(image)), obj(image))#torch.Size([128, 10])
# print(obj(torch.rand(128,1,28,28)))





# x_filtered = CayleyFilter(order=2, jacobi_iterations=2, in_channels=1, out_channels=3)
# print(x_filtered)





# def glorot(tensor):
#     if tensor is not None:
#         stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
#         tensor.data.uniform_(-stdv, stdv)

# def zeros(tensor):
#     if tensor is not None:
#         tensor.data.fill_(0)  

